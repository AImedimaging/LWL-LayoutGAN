import math

import torch.nn
from torch import nn

from torch_utils import persistence, misc
from training.SimNet import LocalGenerator, RenderNet, MappingNetwork, SimGenerator, MiddleModule


@persistence.persistent_class
class EmaGenerator(nn.Module):
    def __init__(self,
                z_dim = 512,
                w_dim = 512,
                w2_dim = 512,
                img_resolution = 256,
                img_channels = 1,
                bbox_dim = 128,
                single_size = 32,
                mid_size = 64,
                min_feat_size = 8,
                mapping_kwargs = {},
                synthesis_kwargs = {}
    ):
        super().__init__()
        self.G = SimGenerator(
            z_dim=z_dim,
            w_dim=w_dim,
            w2_dim=w2_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            bbox_dim=bbox_dim,
            single_size=single_size,
            mid_size=mid_size,
            min_feat_size=min_feat_size,
            mapping_kwargs=mapping_kwargs,
            synthesis_kwargs=synthesis_kwargs
        )
        self.SR = MiddleModule(
            img_resolution=img_resolution,
            img_channels=img_channels,
            channel_base=synthesis_kwargs['channel_base'],
            channel_max=synthesis_kwargs['channel_max']
        )

    def forward(self, z, bbox):
        img, mask, ws1, ws2, mid_mask = self.G(z, bbox)
        img = self.SR(img)
        return img, mask, ws1, ws2, mid_mask
#----------------------------------------------------------------------------





