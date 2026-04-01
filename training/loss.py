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
from torch_utils.custom_ops import img_resampler
from torch_utils.ops import conv2d_gradfix
import torch.nn.functional as F

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_mask, real_bbox, gen_z, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, P, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
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
            img, mask, mask2, ws, mid_mask = self.G(z, bbox)
        return img, mask, mask2, ws, mid_mask

    # def run_Dsr(self, img, sync):
    #     with misc.ddp_sync(self.Dsr, sync):
    #         img = self.Dsr(img)
    #
    #     return img

    def run_D(self, img, mask, bbox, sync, samples=None):

        # if samples is None:
        #     samples = img_resampler(img, bbox, resample_num=16)

        if self.augment_pipe is not None:
            img, mask = self.augment_pipe(img, mask)
            # samples = self.augment_pipe(samples)

        # with misc.ddp_sync(self.Ds, sync):
        #     d2 = self.Ds(samples, bbox, mask)

        with misc.ddp_sync(self.D, sync):
            d1, d2 = self.D(img, bbox, mask)

        return d1, d2


    def accumulate_gradients(self, phase, real_img, real_mask, real_bbox, gen_z, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        # if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG

        lamb = 1.0
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, gen_mask, gen_mask2, _gen_ws, gen_mid_mask = self.run_G(gen_z, real_bbox, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits1, gen_logits2 = self.run_D(gen_img, gen_mask, real_bbox, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits1)
                training_stats.report('Loss/signs/fake', gen_logits1.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits1).mean() + torch.nn.functional.softplus(-gen_logits2).mean()
                training_stats.report('Loss/G/loss', loss_Gmain)
                mask_loss = self.P(gen_mid_mask, F.adaptive_avg_pool2d(real_mask, gen_mid_mask.shape[2:4])) + self.P(gen_mask, real_mask) + self.P(gen_mask2, real_mask)
                training_stats.report('Loss/maskloss', mask_loss)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + mask_loss.mean()).mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_mask, gen_mask2, gen_ws, _gen_mid_mask = self. run_G(gen_z[:batch_size], real_bbox[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                pl_noise2 = torch.randn_like(gen_mask) / np.sqrt(gen_mask.shape[2] * gen_mask.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum(), (gen_mask * pl_noise2).sum(), (gen_mask2 * pl_noise2).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl.square().sum(2).mean(1)
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + gen_mask[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, gen_mask, gen_mask2, _gen_ws, _gen_mid_mask = self.run_G(gen_z, real_bbox, sync=False)
                # gen_img = self.run_Dsr(gen_img, sync=False)
                gen_logits1, gen_logits2 = self.run_D(gen_img, gen_mask, real_bbox, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits1)
                training_stats.report('Loss/signs/fake', gen_logits1.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits1).mean() + torch.nn.functional.softplus(gen_logits2).mean() # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_mask_tmp = real_mask.detach().requires_grad_(do_Dr1)
                real_logits1, real_logits2 = self.run_D(real_img_tmp, real_mask_tmp, real_bbox, sync=sync)
                training_stats.report('Loss/scores/real', real_logits1)
                training_stats.report('Loss/signs/real', real_logits1.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits1).mean() + torch.nn.functional.softplus(-real_logits2).mean() # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_img = torch.autograd.grad(outputs=[real_logits1.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_img.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits1*0 + real_logits2*0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
