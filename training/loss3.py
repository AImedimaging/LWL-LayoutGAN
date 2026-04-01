# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats, misc
from torch_utils.custom_ops import bbox_mask
from torch_utils.ops import upfirdn2d


class Loss:
    def accumulate_gradients(self, phase, real_img, real_mask, real_bbox, gen_z, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()


class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, LPIPS, blur_init_sigma=0, blur_fade_kimg=0, **kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.lpips_loss = LPIPS

    def run_G(self, z, bbox, update_emas=False ):
        img, mask, feats, mid_mask = self.G(z, bbox)
        return img, mask, feats, mid_mask

    def run_D(self, img, mask, bbox, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits, obj_logits, m_logits, m_obj_logits, d_feat = self.D(img, bbox, mask)
        return logits, obj_logits, m_logits, m_obj_logits, d_feat

    def calc_mask_loss(self, real_mask, gen_mask, gen_mid_mask):
        loss1 = self.lpips_loss(real_mask, gen_mask) + F.l1_loss(real_mask, gen_mask)
        # loss2 = self.lpips_loss(real_mask, gen_mask2) + F.l1_loss(real_mask, gen_mask2)
        rs_real_mask = F.adaptive_avg_pool2d(real_mask, gen_mid_mask.shape[2:4])
        loss3 = self.lpips_loss(rs_real_mask, gen_mid_mask) + F.l1_loss(rs_real_mask, gen_mid_mask)
        return loss1 + loss3

    def numeric_sem_mask(self, bbox, mask):
        b_mask = bbox_mask(self.device, bbox[:, :, 1:], mask.shape[2], mask.shape[3])
        # mask = mask * 0.5 + 0.5  # [-1, 1] to [0, 1]
        mask = b_mask * (mask * 0.5 + 0.5)  # [-1, 1] to [0, 1]
        B, ch = bbox.shape[:2]
        weight = torch.arange(ch, 0, -1).to(self.device).unsqueeze(0).repeat([B, 1]).view(B, ch, 1, 1)
        wmask = mask * weight
        wmask = torch.max(wmask, dim=1).values.unsqueeze(1)
        return wmask

    def getCLIPloss(self, vec1, vec2):
        # print(vec1.shape, vec2.shape)
        vec1 = vec1 / (vec1.norm(dim=-1, keepdim=True) + 1e-8)
        vec2 = vec2 / (vec2.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = vec1 @ vec2.T
        # print(similarity)
        batch_size = vec1.shape[0]
        labels = torch.arange(batch_size, device=self.device).long()
        total_loss = (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.t(), labels)) / 2
        # print(total_loss)
        return total_loss

    def accumulate_gradients(self, phase, real_img, real_mask, real_bbox, gen_z, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG
        max_nimg = 50000
        lamb_img = 0.1
        lamb_obj = 1.0
        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0
        lamb = 10
        # real_mask = self.numeric_sem_mask(real_bbox, real_mask)
        real_feats = None
        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, gen_mask, feats, gen_mid_mask = self.run_G(gen_z, real_bbox)
                # gen_mask = self.numeric_sem_mask(real_bbox, gen_mask)
                gen_logits, gen_obj_logits, gen_m_logits, gen_m_obj_logits, d_feat = self.run_D(gen_img, gen_mask, real_bbox, blur_sigma=blur_sigma)
                loss_Gmain = lamb_img*(-gen_logits).mean() + lamb_obj*(-gen_obj_logits).mean()
                loss_Gmain2 = lamb_img*(-gen_m_logits).mean() + lamb_obj*(-gen_m_obj_logits).mean()
                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

                # img_loss = self.lpips_loss(real_img, gen_img)
                mask_loss = self.calc_mask_loss(real_mask, gen_mask, gen_mid_mask)
                img_rec_loss = 0
                img_rec_loss = self.lpips_loss(real_img, gen_img) + F.l1_loss(real_img, gen_img)
                r1, r2 = 0, 0
                r1 = self.getCLIPloss(gen_obj_logits.detach(), gen_m_obj_logits) + self.getCLIPloss(gen_logits.detach(), gen_m_logits)
                # r2 = min((max_nimg/(cur_nimg+1e-8)), 10)* self.get_ggdr_reg(feats, d_feat.detach())

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + loss_Gmain2 + mask_loss + img_rec_loss - r1 + r2).backward()


        if do_Dmain:

            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, gen_mask, feats, gen_mid_mask = self.run_G(gen_z, real_bbox, update_emas=True)
                # gen_mask = self.numeric_sem_mask(real_bbox, gen_mask)
                gen_logits, gen_obj_logits, gen_m_logits, gen_m_obj_logits, d_feat = self.run_D(gen_img, gen_mask, real_bbox, blur_sigma=blur_sigma)
                loss_Dgen = lamb_img * (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean() + lamb_obj * (F.relu(torch.ones_like(gen_obj_logits) + gen_obj_logits)).mean()
                loss_Dgen2 = lamb_img * (F.relu(torch.ones_like(gen_m_logits) + gen_m_logits)).mean() + lamb_obj * (F.relu(torch.ones_like(gen_m_obj_logits) + gen_m_obj_logits)).mean()
                r1 = 0
                r1 = lamb_img * (1 + self.getCLIPloss(gen_logits.detach(), gen_m_logits)).mean() + lamb_obj * (1 + self.getCLIPloss(gen_obj_logits.detach(), gen_m_obj_logits)).mean()
                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen + loss_Dgen2 + r1).backward()

            # Dmain: Maximize logits for real images.
            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                real_mask_tmp = real_mask.detach().requires_grad_(False)
                real_bbox_tmp = real_bbox.detach().requires_grad_(False)
                real_logits, real_obj_logits, real_m_logits, real_m_obj_logits, d_feat = self.run_D(real_img_tmp, real_mask_tmp, real_bbox_tmp, blur_sigma=blur_sigma)
                loss_Dreal = lamb_img * (F.relu(torch.ones_like(real_logits) - real_logits)).mean() + lamb_obj * (F.relu(torch.ones_like(real_obj_logits) - real_obj_logits)).mean()
                loss_Dreal2 = lamb_img * (F.relu(torch.ones_like(real_m_logits) - real_m_logits)).mean() + lamb_obj * (F.relu(torch.ones_like(real_m_obj_logits) - real_m_obj_logits)).mean()
                r1 = 0
                # r1 = lamb_img * (1 - self.getCLIPloss(real_logits.detach(), real_m_logits)).mean() + lamb_obj * (
                #             1 - self.getCLIPloss(real_obj_logits.detach(), real_m_obj_logits)).mean()
                r1 = self.getCLIPloss(real_obj_logits.detach(), real_m_obj_logits) + self.getCLIPloss(real_logits.detach(), real_m_logits)
                # Logging
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            with torch.autograd.profiler.record_function('Dreal_backward'):
                (loss_Dreal + loss_Dreal2 + r1).backward()

    def cosine_distance(self, x, y):
        return 1. - F.cosine_similarity(x, y).mean()

    def get_ggdr_reg(self, x, y):
        loss_gen_recon = 0

        loss_gen_recon = 10 * self.cosine_distance(x, y)

        return loss_gen_recon