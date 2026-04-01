# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.aug_util import ReversibleAugment
from torch_utils.ops import conv2d_gradfix
import torch.nn.functional as F
#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_mask, real_bbox, gen_z, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G,  D, P, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.P = P
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def run_G(self, z, bbox, sync):
        with misc.ddp_sync(self.G, sync):
            img, mask, ws, ws2, mid_masks = self.G(z, bbox)
        return img, mask, ws, ws2, mid_masks

    def run_D(self, img, mask, sync):
        if self.augment_pipe is not None:
            img, mask = self.augment_pipe(img, mask)
        with misc.ddp_sync(self.D, sync):
            vecs, logits = self.D(img, mask, getVec=True)
        return vecs, logits

    def compute_simloss(self, vec1, vec2, t=0.3):
        vec1 = vec1 / torch.norm(vec1, dim=-1, keepdim=True)
        vec2 = vec2 / torch.norm(vec2, dim=-1, keepdim=True)
        similarity = torch.nn.functional.softmax(torch.mm(vec1, vec2.T)/t, dim=0)
        # print(-torch.log2(similarity.diag()))
        return similarity

    def accumulate_gradients(self, phase, real_img, real_mask, real_bbox, gen_z, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        mask_loss = 0
        # Gmain: Maximize logits for generated images.

        aug_real_img, aug_real_mask, aug_real_bbox, aug_ops = ReversibleAugment(real_img, real_mask, real_bbox)

        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, gen_mask, _, _, _ = self.run_G(gen_z, real_bbox, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                mask_loss = self.P(gen_img, real_img) + self.P(gen_mask, real_mask)
                gen_vecs, gen_logits = self.run_D(gen_img, gen_mask, sync=False)

                gen_a_img, gen_a_mask, _, _, _ = self.run_G(gen_z, aug_real_bbox, sync=(sync and not do_Gpl))  # May get synced by Gpl.
                gen_a_vecs, agen_a_logits = self.run_D(gen_a_img, gen_a_mask, sync=False)
                gen_a_vecs = gen_a_vecs.detach()

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)

                simloss = 0
                similarity = self.compute_simloss(gen_vecs, gen_a_vecs)
                simloss = -torch.log2(similarity.diag())

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (gen_mask[:, 0, 0, 0]*0 + loss_Gmain + mask_loss + simloss).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_mask, gen_ws, gen_ws2, _gen_mid_masks = self.run_G(gen_z[:batch_size], real_bbox[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])

                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_ws, pl_ws2 = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws, gen_ws2], create_graph=True, only_inputs=True)[:2]
                pl_ws_lengths = torch.cat([pl_ws, pl_ws2], dim=1).square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_ws_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_ws_lengths - pl_mean).square()
                training_stats.report('Loss/ws/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + gen_mask[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, gen_mask, _gen_ws, _gen_ws2, gen_mid_masks = self.run_G(gen_z, real_bbox, sync=False)

                gen_vecs, gen_logits = self.run_D(gen_img, gen_mask, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                gen_a_img, gen_a_mask, _, _, _ = self.run_G(gen_z, aug_real_bbox, sync=False)  # May get synced by Gpl.
                gen_a_vecs, gen_a_logits = self.run_D(gen_a_img, gen_a_mask, sync=False)  # Gets synced by loss_Dreal.
                gen_a_vecs = gen_a_vecs.detach()
                gen_a_logits = gen_a_logits.detach()

                simloss = 0
                similarity = self.compute_simloss(gen_vecs, gen_a_vecs)
                simloss = -torch.log2(1.0 - similarity.diag())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen + simloss).mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_mask_tmp = real_mask.detach().requires_grad_(do_Dr1)

                real_vecs, real_logits = self.run_D(real_img_tmp, real_mask_tmp, sync=sync)
                real_a_vecs, real_a_logits = self.run_D(aug_real_img, aug_real_mask, sync=sync)
                real_a_vecs = real_a_vecs.detach()

                simloss = 0
                similarity = self.compute_simloss(real_vecs, real_a_vecs)
                simloss = -torch.log2(similarity.diag())


                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        # r1_img, r1_mask = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp, real_mask_tmp], create_graph=True, only_inputs=True)
                        r1_img = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_img_penalty = r1_img.square().sum([1,2,3])
                    training_stats.report('Loss/r1_img_penalty', r1_img_penalty)

                    r1_mask_penalty = 0
                    # r1_mask_penalty = r1_img.square().sum([1, 2, 3])
                    # training_stats.report('Loss/r1_mask_penalty', r1_mask_penalty)

                    loss_Dr1 = r1_img_penalty * (self.r1_gamma / 2) + r1_mask_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + simloss + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
