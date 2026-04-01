import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch_utils import persistence, misc
from torch_utils.common import tensor_shift
from torch_utils.custom_ops import _boxes_to_grid, img_resampler
from torch_utils.ops import grid_sample_gradfix
from training.networks import SynthesisBlock, normalize_2nd_moment, FullyConnectedLayer, \
    Conv2dLayer, DiscriminatorBlock, DiscriminatorEpilogue, MinibatchStdLayer, ToRGBLayer
#       from training.roi_layers import ROIAlign
from torchvision.ops import RoIAlign as ROIAlign

@persistence.persistent_class
class PositionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, N_freqs=10, logscale=True):
        super(PositionEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = out_channels
        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

        self.weights = nn.Parameter(torch.randn(out_channels, in_channels * (len(self.funcs) * N_freqs + 1), 1, 1))
        self.scale = 1 / math.sqrt(in_channels * (len(self.funcs) * N_freqs + 1))

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        out = torch.cat(out, 1)
        out = F.conv2d(out, self.scale * self.weights, bias=None)
        return out

@persistence.persistent_class
class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x

@persistence.persistent_class
class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv2dLayer(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.qkv = torch.nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.attention = QKVAttention()
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(x)
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])

@persistence.persistent_class
class SegEncoder(nn.Module):
    def __init__(self, out_channel, in_channel=128, conv_clamp=None, resample_filter=[1, 3, 3, 1], channels_last=False):
        super().__init__()
        kernel_size = 3
        nhidden = out_channel
        self.mlp_shared = Conv2dLayer(in_channel, nhidden, kernel_size=kernel_size, activation='lrelu', conv_clamp=None, resample_filter=resample_filter)
        self.mlp_gamma = Conv2dLayer(nhidden, out_channel, kernel_size=kernel_size, conv_clamp=conv_clamp, resample_filter=resample_filter)
        self.mlp_beta = Conv2dLayer(nhidden, out_channel, kernel_size=kernel_size, conv_clamp=conv_clamp, resample_filter=resample_filter)
        # self.blur = Blur(blur_kernel, pad=(2, 1))

    def forward(self, style_img, size=(64, 64), shift=None):
        # style_img = scatter(style_img)
        # style_img = F.interpolate(style_img, size=size, mode='bilinear', align_corners=True)

        actv = self.mlp_shared(style_img)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # gamma, beta = self.blur(gamma), self.blur(beta)
        if shift is not None:
            height, width = size
            gamma, beta = tensor_shift(gamma, int(shift[0] * width / 512), int(shift[1] * height / 512)), \
                          tensor_shift(beta, int(shift[0] * width / 512), int(shift[1] * height / 512))

        return gamma, beta


@persistence.persistent_class
class LocalGenerator(nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output image resolution.
                 img_channels,  # Number of color channels.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 **block_kwargs,  # Arguments for SynthesisBlock.
                 ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {4:512, 8:256, 16:256, 32:128}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0

        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                                   img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv

            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

            # if not is_last:
            #     attn1 = AttentionBlock(out_channels, 4)
            #     setattr(self, f'at1{res}', attn1)
            #     attn2 = AttentionBlock(out_channels, 4)
            #     setattr(self, f'at2{res}', attn2)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img, _ = block(x, img, None, ws=cur_ws, **block_kwargs)

            # dtype = x.dtype
            # x = x.to(torch.float32)
            # if res < self.img_resolution:
            #     attn1 = getattr(self, f'at1{res}')
            #     attn2 = getattr(self, f'at2{res}')
            #     x = attn2(attn1(x))
            #     x = x.to(dtype=dtype)
        return img


@persistence.persistent_class
class RenderNet(nn.Module):
    def __init__(self,
                 w_dim,
                 in_resolution,  # Output image resolution.
                 img_resolution,  # Output image resolution
                 img_channels,  # Number of color channels.
                 mask_channels,  # Number of color channels.
                 mid_size,  # Number of color channels.
                 mid_channels,  # Number of color channels.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 **block_kwargs,  # Arguments for SynthesisBlock.
                 ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.in_resolution = in_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.in_resolution_log2 = int(np.log2(in_resolution))
        self.img_channels = img_channels
        self.mask_channels = mask_channels
        self.mid_size = mid_size
        self.mid_channels= mid_channels
        self.block_resolutions = [2 ** i for i in range(self.in_resolution_log2+1, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in [in_resolution] + self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.convert = Conv2dLayer(in_channels=self.mask_channels, out_channels=512, kernel_size=1)
        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res//2 != mid_size else channels_dict[res // 2] + mid_channels
            # in_channels = channels_dict[res // 2]
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)

            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                                   img_channels=img_channels, mask_channels=1, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

            # if res <= 32:
            #     tblock = TransformerBlock(out_channels, out_channels, num_heads=4, num_layers=3)
            #     setattr(self, f'tb{res}', tblock)

            # if res < self.mid_size:
            #     fusion = SegEncoder(out_channel=out_channels, in_channel=self.mask_channels)
            #     setattr(self, f'fus{res}', fusion)



    def forward(self, x, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x_orig, x = x, F.adaptive_avg_pool2d(x, (self.in_resolution, self.in_resolution))

        img = mask = None
        x = self.convert(x)
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            if res // 2 == self.mid_size:
                x = torch.cat([x, x_orig], dim=1)

            block = getattr(self, f'b{res}')
            x, img, mask = block(x, img, mask, cur_ws, **block_kwargs)
            # dtype = x.dtype
            # if res <= 32:
            #     x = x.to(torch.float32)
            #     tblock = getattr(self, f'tb{res}')
            #     x = tblock(x).to(dtype)
        return img, mask


@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        w_dim,                      # Intermediate latent (W) dimensionality.
        # w2_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_ws2,                    # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        # self.w2_dim = w2_dim
        self.num_ws = num_ws
        self.num_ws2 = num_ws2
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim] + [layer_features] * (num_layers - 1) + [w_dim]

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.

        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)


        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws + self.num_ws2, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x[:, :self.num_ws], x[:, self.num_ws:]

#----------------------------------------------------------------------------

@persistence.persistent_class
class SimGenerator(nn.Module):
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
        assert mid_size < img_resolution
        assert min_feat_size < mid_size and mid_size % min_feat_size == 0
        self.img_resolution = img_resolution
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.w_dim = w_dim
        self.bbox_dim = bbox_dim
        self.log_size = int(math.log(img_resolution, 2))
        self.img_channels = img_channels
        self.mid_size = mid_size
        self.min_feat_size = min_feat_size
        self.single_size = single_size
        self.gen_single = LocalGenerator(w_dim=w_dim, img_resolution=self.single_size, img_channels=1, **synthesis_kwargs)
        self.num_ws = self.gen_single.num_ws
        self.render_net = RenderNet(w_dim=w_dim,  in_resolution=min_feat_size, img_resolution=img_resolution, img_channels=img_channels, mask_channels=self.bbox_dim, mid_size=mid_size, mid_channels=bbox_dim, **synthesis_kwargs)
        self.num_ws2 = self.render_net.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_ws=self.num_ws,  num_ws2=self.num_ws2, **mapping_kwargs)

        # self.SRnet = MiddleModule(img_resolution=img_resolution, img_channels=img_channels)

    def forward(self, z, bbox, truncation_psi=1, truncation_cutoff=None):
        misc.assert_shape(bbox, [None, self.bbox_dim, 5])
        labels, bbox = bbox[:, :, 0], bbox[:, :, 1:]
        ws1, ws2 = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        sin_m = (self.gen_single(ws1) + 1.0) / 2.0 # adjust from [-1, 1] to [0, 1]
        sin_ms = sin_m.repeat(1, self.bbox_dim, 1, 1).view(-1,  1,  self.single_size, self.single_size)
        grid = _boxes_to_grid(bbox.view(-1, 4), self.mid_size, self.mid_size).to(sin_ms.dtype)
        mid_masks = grid_sample_gradfix.grid_sample(sin_ms, grid).view(-1, self.bbox_dim, self.mid_size, self.mid_size)
        # mid_masks = mid_masks.mul(2.0) - 1.0 # [0, 1]adjust to [-1, 1]
        img, mask = self.render_net(mid_masks.mul(2.0) - 1.0, ws2)
        # img = self.SRnet(img)
        return img, mask, ws1, ws2, mid_masks.sum(dim=1, keepdim=True).clamp(max=1).mul(2.0) - 1.0

#----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=1, down=1):
        super().__init__()
        self.conv0 = Conv2dLayer(in_channels, out_channels, kernel_size=3, up=up, down=down)
        self.conv1 = Conv2dLayer(out_channels, out_channels, kernel_size=3)
        self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, up=up, down=down)

    def forward(self, x):
        return self.skip(x, gain=np.sqrt(0.5)) + self.conv1(self.conv0(x), gain=np.sqrt(0.5))


@persistence.persistent_class
class MiddleModule(torch.nn.Module):
    def __init__(self,
                 img_resolution,
                 img_channels,
                 channel_base = 32768,
                 channel_max = 512,
                 ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.middle_resolution = 64
        self.channel_base = channel_base
        self.channel_max = channel_max
        self.img_resolution_log2 = int(np.log2(self.img_resolution))
        self.middle_resolution_log2 = int(np.log2(self.middle_resolution))
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, self.middle_resolution_log2-1, -1)]
        channels_dict = {res: min(self.channel_base // res, self.channel_max) for res in self.block_resolutions + [self.middle_resolution]}
        self.down = nn.ModuleList([Conv2dLayer(self.img_channels, channels_dict[self.img_resolution], kernel_size=3)])

        for res in self.block_resolutions[:-1:]:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            self.down.append(ResBlock(in_channels, out_channels, down=2))

        self.middle = ResBlock(channels_dict[self.middle_resolution], channels_dict[self.middle_resolution])

        self.up = nn.ModuleList([])

        for res in self.block_resolutions[:0:-1]:
            in_channels = channels_dict[res] * 2
            out_channels = channels_dict[res * 2]
            self.up.append(ResBlock(in_channels, out_channels, up=2))


        self.out = nn.Sequential(
            Conv2dLayer(channels_dict[self.img_resolution]*2, channels_dict[self.img_resolution], kernel_size=1),
            ToRGBLayer(channels_dict[self.img_resolution], self.img_channels)
        )

    def forward(self, x):

        input_map = []
        for net in self.down:
            x = net(x)
            input_map.append(x)

        x = self.middle(x)
        for net in self.up:
            x = torch.cat([x, input_map.pop(-1)], dim=1)
            x = net(x)

        x = torch.cat([x, input_map.pop(-1)], dim=1)

        x = self.out(x)
        return x

#----------------------------------------------------------------------------
@persistence.persistent_class
class DualDiscriminator(torch.nn.Module):
    def __init__(self,
        img_resolution      = 256,                 # Input resolution.
        img_channels        = 3,                   # Number of input color channels.
        mask_channels       = 1,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        single                = False,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.sin_img_resolution_log2 = int(np.log2(32))
        self.img_channels = img_channels
        self.mask_channels = mask_channels
        self.single = single
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        self.sin_block_resolutions = [2 ** i for i in range(self.sin_img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)


        common_kwargs = dict(architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)

            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res, img_channels=img_channels,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)

            if res in [64, 32, 16]:
                obj_feat = ResBlock(in_channels, out_channels)
                setattr(self, f'obj_feat{res}', obj_feat)

            cur_layer_idx += block.num_layers

        self.obj_feat8 = ResBlock(channels_dict[8], channels_dict[4], down=2)

        #self.roi_align_s = ROIAlign((8, 8), 1.0 / 4.0, int(0))
        #self.roi_align_m = ROIAlign((8, 8), 1.0 / 8.0, int(0))
        #self.roi_align_l = ROIAlign((8, 8), 1.0 / 16.0, int(0))

        self.b4 = DiscriminatorEpilogue(channels_dict[4], resolution=4, getVec=False, **epilogue_kwargs, **common_kwargs)
        self.obj_b4 = DiscriminatorEpilogue(channels_dict[4], resolution=4, getVec=False, **epilogue_kwargs, **common_kwargs)


        # self.out = FullyConnectedLayer(channels_dict[4], 1)
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


    def forward(self, img, bbox, mask=None, condition=None, **block_kwargs):
        x = None
        y = None

        B = img.shape[0]

        label, bbox = self.pre_bbox(self.xywh2x0y0x1y1(bbox))

        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 64) * ((bbox[:, 4] - bbox[:, 2]) < 64)
        bbox_s, bbox_ns = bbox[s_idx], bbox[~s_idx]
        label_s, label_ns = label[s_idx], label[~s_idx]
        m_idx = ((bbox_ns[:, 3] - bbox_ns[:, 1]) < 128) * ((bbox_ns[:, 4] - bbox_ns[:, 2]) < 128)
        bbox_m, bbox_l = bbox_ns[m_idx], bbox_ns[~m_idx]
        label_m, label_l = label_ns[m_idx], label_ns[~m_idx]
        feat32, feat16, feat8 = None, None, None

        feats = []
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img, feat = block(x, img, **block_kwargs)

            if res == 128:
                feat32 = x.to(torch.float32) # 64 * 64
            if res == 64:
                feat16 = x.to(torch.float32) # 32 * 32
            if res == 32:
                feat8 = x.to(torch.float32) # 16 * 16

            [feats.append(f) for f in feat]

        feat_s = self.obj_feat64(feat32)
        feat_s = self.obj_feat32(feat_s)
        feat_s = self.obj_feat16(feat_s)
        feat_s = torchvision.ops.roi_align(feat_s, bbox_s, (8, 8), 1.0 / 4.0, int(0))

        feat_m = self.obj_feat32(feat16)
        feat_m = self.obj_feat16(feat_m)
        feat_m = torchvision.ops.roi_align(feat_m, bbox_m, (8, 8), 1.0 / 8.0, int(0))

        feat_l = self.obj_feat16(feat8)
        feat_l = torchvision.ops.roi_align(feat_l, bbox_l, (8, 8), 1.0 / 16.0, int(0))

        obj_feat = torch.cat([feat_s, feat_m, feat_l], dim=0)
        obj_feat = self.obj_feat8(obj_feat)

        y = self.obj_b4(obj_feat).view(B, -1).mean(1, keepdim=True)
        x = self.b4(x)


        return x, y

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        img_resolution      = 256,                 # Input resolution.
        img_channels        = 3,                   # Number of input color channels.
        mask_channels       = 1,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        # self.SRnet = MiddleModule(img_resolution=img_resolution, img_channels=img_channels)
        self.Mnet = DualDiscriminator(
            img_resolution=img_resolution,
            img_channels=img_channels,
            mask_channels=mask_channels,
            architecture=architecture,
            channel_base=channel_base,
            channel_max=channel_max,
            num_fp16_res=num_fp16_res,
            conv_clamp=conv_clamp,
            block_kwargs=block_kwargs,
            mapping_kwargs=mapping_kwargs,
            epilogue_kwargs=epilogue_kwargs
        )

        # self.Snet = DualDiscriminator(
        #     img_resolution=32,
        #     img_channels=img_channels,
        #     mask_channels=mask_channels,
        #     architecture=architecture,
        #     channel_base=channel_base,
        #     channel_max=channel_max,
        #     num_fp16_res=num_fp16_res,
        #     conv_clamp=conv_clamp,
        #     single=True,
        #     block_kwargs=block_kwargs,
        #     mapping_kwargs=mapping_kwargs,
        #     epilogue_kwargs=epilogue_kwargs
        # )

    def bbox_resampler(self, bbox, sample_num=8):
        B= bbox.shape[0]
        bi, ni = torch.where((bbox[:, :, 2] * bbox[:, :, 3]) > 0)
        bc = np.array([len(bi[bi == b]) for b in range(B)])  # batch count
        # rs = torch.cat([torch.randint(bc[:i].sum(), bc[:i+1].sum(), [resample_num]) for i in range(B)]) # resample
        rs = []
        for i in range(B):
            tmp = [num % bc[i] for num in range(sample_num)] if bc[i] > 0 else [0 for _ in range(sample_num)]
            rs.append(tmp)
        rs = np.stack(rs)
        bbox2 = torch.zeros([B, sample_num, 5]).to(bbox.device)
        for i, (box, r) in enumerate(zip(bbox, rs)):
            bbox2[i] = box[r]

        return bbox2

    def forward(self, img, bbox, mask, **block_kwargs):
        # img = self.SRnet(img)
        sample_bbox = self.bbox_resampler(bbox)
        d1, d2 = self.Mnet(img, sample_bbox, mask = mask, **block_kwargs)
        # samples = img_resampler(img, bbox, resample_num=16) # (B*16, C, H, W)
        # d2 = self.Snet(img=samples, **block_kwargs).view(img.shape[0], 16, 1).mean(1)
        # return d1, d2
        return d1, d2

if __name__ == "__main__":
    pass
