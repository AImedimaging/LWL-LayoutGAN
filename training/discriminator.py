from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from training.blocks import DownBlock, DownBlockPatch, conv2d, LSBlock
from training.networks import Conv2dLayer
from training.projector import F_RandomProj
from training.diffaug import DiffAugment

CHANNEL_DICT = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}
class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False):
        super().__init__()
        channel_dict = CHANNEL_DICT

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        self.layers = nn.ModuleList([])

        # Head if the initial input is the full modality
        if head:
            self.layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            self.layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2

        self.layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        # self.main = nn.Sequential(*layers)
        # self.main = nn.ModuleList(*layers)

    def forward(self, x):
        # return self.main(x)
        feat64 = None
        feat32 = None
        feat16 = None
        for layer in self.layers:
            if x.shape[2] == 64:
                feat64 = x
            if x.shape[2] == 32:
                feat32 = x
            if x.shape[2] == 16:
                feat16 = x
            x = layer(x)

        return x, feat64, feat32, feat16

class SingleDiscCond(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False, c_dim=1000, cmap_dim=64, embedding_dim=128):
        super().__init__()
        self.cmap_dim = cmap_dim

        # midas channels
        channel_dict = CHANNEL_DICT

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)

        # additions for conditioning on class information
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
        self.embed = nn.Embedding(num_embeddings=c_dim, embedding_dim=embedding_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(self.embed.embedding_dim, self.cmap_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)

        # conditioning via projection
        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=1,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        separable=False,
        patch=False,
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc

        num64 = 0
        num32 = 0
        num16 = 0
        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            if start_sz >= 64:
                num64 += 1
            if start_sz >= 32:
                num32 += 1
            if start_sz >= 16:
                num16 += 1
            mini_discs.append(Disc(nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch))

        self.mini_discs = nn.ModuleList(mini_discs)


        self.lsb1 = LSBlock(CHANNEL_DICT[64]*num64, CHANNEL_DICT[32], separable=separable)
        self.lsb2 = LSBlock(CHANNEL_DICT[32]*num32, CHANNEL_DICT[32], separable=separable)
        self.lsb3 = LSBlock(CHANNEL_DICT[16]*num16, CHANNEL_DICT[32], separable=separable)

        self.last = conv2d(CHANNEL_DICT[32], 1, 4, 1, 0, bias=False)

    def forward(self, features, bbox_s, bbox_m, bbox_l):
        all_logits = []
        all_feat64 = []
        all_feat32 = []
        all_feat16 = []
        batch = 1
        for k, disc in enumerate(self.mini_discs):
            logits, feat64, feat32, feat16 = disc(features[str(k)])
            batch = features[str(k)].size(0)
            # all_logits.append(logits.view(features[str(k)].size(0), -1))
            all_logits.append(logits)

            if feat64 is not None:
                all_feat64.append(feat64)
            if feat32 is not None:
                all_feat32.append(feat32)
            if feat16 is not None:
                all_feat16.append(feat16)
        all_logits = torch.cat(all_logits, dim=1)
        all_feat64 = torch.cat(all_feat64, dim=1)
        all_feat32 = torch.cat(all_feat32, dim=1)
        all_feat16 = torch.cat(all_feat16, dim=1)
        # print(all_feat64.shape, all_feat32.shape, all_feat16.shape)
        all_feat64, all_feat32, all_feat16 = self.lsb1(all_feat64), self.lsb2(all_feat32), self.lsb3(all_feat16)

        all_feat_s = roi_align(all_feat64, bbox_s, (8, 8))
        all_feat_m = roi_align(all_feat32, bbox_m, (8, 8))
        all_feat_l = roi_align(all_feat16, bbox_l, (8, 8))

        feats = torch.cat([all_feat_s, all_feat_m, all_feat_l], dim=0)
        bbox = torch.cat([bbox_s, bbox_m, bbox_l], dim=0)
        all_feats = None
        for i in range(batch):
            idx = bbox[:, 0] == i
            tmp = feats[idx].mean(dim=0, keepdim=True)
            all_feats = tmp if all_feats is None else torch.cat([all_feats, tmp], dim=0)

        # all_feat = torch.cat([all_feat64.mean(dim=0, keepdim=True), all_feat32.mean(dim=0, keepdim=True), all_feat16.mean(dim=0, keepdim=True)], dim=0)
        # all_feat_logit = self.last(all_feat)

        return all_logits, self.last(all_feats)

class ProjectedDiscriminator(torch.nn.Module):
    def __init__(
        self,
        diffaug=True,
        interp224=True,
        mask_channels = 256,
        img_resolution = 256,
        backbone_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.diffaug = diffaug
        self.interp224 = interp224
        self.mask_channels = mask_channels
        self.img_resolution = img_resolution
        self.feature_network = F_RandomProj(**backbone_kwargs)

        self.discriminator1 = MultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            **backbone_kwargs,
        )
        self.discriminator2 = MultiScaleD(
            channels=self.feature_network.CHANNELS,
            resolutions=self.feature_network.RESOLUTIONS,
            **backbone_kwargs,
        )

        # self.final_conv_I = Conv2dLayer(in_channels=4,  out_channels=1, kernel_size=3)
        # self.final_conv_O = Conv2dLayer(in_channels=3,  out_channels=1, kernel_size=3)

    def train(self, mode=True):
        self.feature_network = self.feature_network.train(False)
        self.discriminator1 = self.discriminator1.train(mode)
        self.discriminator2 = self.discriminator2.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def xywh2x0y0x1y1(self, bbox1):
        bbox = bbox1.clone()
        bbox[:, :, 1] = bbox1[:, :, 1] - bbox1[:, :, 3]/2
        bbox[:, :, 2] = bbox1[:, :, 2] - bbox1[:, :, 4]/2
        bbox[:, :, 3] = bbox1[:, :, 1] + bbox1[:, :, 3]
        bbox[:, :, 4] = bbox1[:, :, 2] + bbox1[:, :, 4]
        return bbox

    def pre_bbox(self, bbox):
        bbox[:, :, 1:] = (bbox[:, :, 1:] * (self.img_resolution-1)).clamp(0, self.img_resolution-1).to(torch.uint8)
        label, bbox = bbox[:, :, 0], bbox[:, :, 1:]
        idx = torch.arange(start=0, end=bbox.size(0), device=bbox.device).view(bbox.size(0), 1, 1).expand(-1, bbox.size(1), -1).float()
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        return label, bbox

    def classifier(self, label, bbox):
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 32) * ((bbox[:, 4] - bbox[:, 2]) < 32)
        bbox_s, bbox_ns = bbox[s_idx], bbox[~s_idx]
        label_s, label_ns = label[s_idx], label[~s_idx]
        m_idx = ((bbox_ns[:, 3] - bbox_ns[:, 1]) < 64) * ((bbox_ns[:, 4] - bbox_ns[:, 2]) < 64)
        bbox_m, bbox_l = bbox_ns[m_idx], bbox_ns[~m_idx]
        label_m, label_l = label_ns[m_idx], label_ns[~m_idx]
        return bbox_s, bbox_m, bbox_l

    def forward(self, x, bbox, mask, beta=0.0):
        bbox = self.xywh2x0y0x1y1(bbox)
        if self.diffaug:
            x, mask, bbox = DiffAugment(x, mask, bbox, policy='color,translation,cutout')

        if self.interp224:
            x = F.interpolate(x, 224, mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, 224, mode='bilinear', align_corners=False)


        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        m_feat = mask.repeat(1, 3, 1, 1)

        x_feat = self.feature_network(x)
        m_feat = self.feature_network(m_feat)


        pre_label, pre_bbox = self.pre_bbox(bbox)
        bbox_s, bbox_m, bbox_l = self.classifier(pre_label, pre_bbox)

        logits, obj_logits = self.discriminator1(x_feat, bbox_s, bbox_m, bbox_l)
        logits, obj_logits = logits.flatten(1, 3), obj_logits.flatten(1, 3)
        m_logits, m_obj_logits = torch.zeros_like(logits).to(logits), torch.zeros_like(obj_logits).to(obj_logits)
        m_logits, m_obj_logits = self.discriminator2(m_feat, bbox_s, bbox_m, bbox_l)
        m_logits, m_obj_logits = m_logits.flatten(1, 3), m_obj_logits.flatten(1, 3)

        # m_logits, m_obj_logits = 0, 0
        # logits = logits + beta * m_logits
        # obj_logits = obj_logits + beta * m_obj_logits

        # logits = self.final_conv_I(logits)
        # obj_logits = self.final_conv_O(obj_logits.squeeze(1).unsqueeze(0))

        return logits, obj_logits, m_logits, m_obj_logits, x_feat['0']