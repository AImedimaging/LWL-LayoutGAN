import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch_utils import persistence, misc
from torch_utils.common import tensor_shift
from torch_utils.custom_ops import _boxes_to_grid, img_resampler, bbox_mask, color_map
from torch_utils.ops import grid_sample_gradfix
from training.blocks import SEBlock, normalization, conv_nd, SiLU, avg_pool_nd, zero_module
from training.networks import SynthesisBlock, normalize_2nd_moment, FullyConnectedLayer, \
    Conv2dLayer, DiscriminatorBlock, DiscriminatorEpilogue, MinibatchStdLayer, ToRGBLayer, SynthesisLayer
from torchvision.ops import RoIAlign as ROIAlign

from training.projector import F_RandomProj


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


def xf_convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: torch.Tensor):
        return super().forward(x.float()).to(x.dtype)


class MultiheadAttention(nn.Module):
    def __init__(self, n_ctx, width, heads):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads, n_ctx)

    def forward(self, x, key_padding_mask=None):
        x = self.c_qkv(x)
        x = self.attention(x, key_padding_mask)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx

    def forward(self, qkv, key_padding_mask=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards

        if key_padding_mask is not None:
            weight = weight.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (N, 1, 1, L1)
                float('-inf'),
            )
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            n_ctx: int,
            width: int,
            heads: int,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx,
            width,
            heads,
        )
        self.ln_1 = LayerNorm(width)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm(width)

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        x = x + self.attn(self.ln_1(x), key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            n_ctx: int,
            width: int,
            layers: int,
            heads: int,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx,
                    width,
                    heads,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        for block in self.resblocks:
            x = block(x, key_padding_mask)
        return x


@persistence.persistent_class
class SegEncoder(nn.Module):
    def __init__(self, out_channel, in_channel=128, conv_clamp=None, resample_filter=[1, 3, 3, 1], channels_last=False):
        super().__init__()
        kernel_size = 3
        nhidden = out_channel
        self.mlp_shared = Conv2dLayer(in_channel, nhidden, kernel_size=kernel_size, activation='lrelu', conv_clamp=None,
                                      resample_filter=resample_filter)
        self.mlp_gamma = Conv2dLayer(nhidden, out_channel, kernel_size=kernel_size, conv_clamp=conv_clamp,
                                     resample_filter=resample_filter)
        self.mlp_beta = Conv2dLayer(nhidden, out_channel, kernel_size=kernel_size, conv_clamp=conv_clamp,
                                    resample_filter=resample_filter)
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
        channels_dict = {4: 512, 8: 256, 16: 256, 32: 128}
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
            x, img = block(x, img, ws=cur_ws, **block_kwargs)

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
        self.mid_channels = mid_channels
        self.block_resolutions = [2 ** i for i in range(self.in_resolution_log2 + 1, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in [in_resolution] + self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.get_feat_res = [16, 32, 64]

        self.convert = Conv2dLayer(in_channels=self.mask_channels, out_channels=512, kernel_size=1)
        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res // 2 != mid_size else channels_dict[res // 2] + mid_channels
            # in_channels = channels_dict[res // 2]
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)

            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                                   img_channels=img_channels, mid_size=mid_size, mask_channels=self.mask_channels,
                                   is_last=is_last, use_fp16=use_fp16, **block_kwargs)

            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

        # decoder_res = 64
        # self.dec_resolutions = [2 ** i for i in range(self.in_resolution_log2 + 1, int(np.log2(decoder_res)) + 1)]
        #
        # for res in self.dec_resolutions:
        #     out_channels = channels_dict[res]
        #     in_channels = channels_dict[res // 2]
        #     if res != self.dec_resolutions[0]:
        #         in_channels *= 2
        #
        #     if res == self.dec_resolutions[-1]:
        #         out_channels = 64
        #
        #     block = Conv2dLayer(in_channels, out_channels, kernel_size=1, activation='linear', up=2)
        #     setattr(self, f'b{res}_dec', block)


    def forward(self, x, ws, bbox, get_feats=False, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x_orig, x = x, F.adaptive_avg_pool2d(x, (self.in_resolution, self.in_resolution))  # [-1, 1]

        img = None
        mask_x = None
        mask_y = None
        mask = x_orig
        feats = {}

        x = self.convert(x)
        for res, cur_ws in zip(self.block_resolutions, block_ws):

            if res // 2 == self.mid_size:
                x = torch.cat([x, x_orig], dim=1)

            block = getattr(self, f'b{res}')
            x, img, mask_x = block(x, img, mask_x, cur_ws, **block_kwargs) # mask_x 通道为 1
            mask_x = bbox_mask(x.device, bbox, res, res).sum(dim=1, keepdim=True).clamp(0, 1) * (mask_x + 1) - 1
            # if res >= self.mid_size:
            #     mask_y = bbox_mask(x.device, bbox, res, res) * (mask_x+1) - 1
            #     cb = getattr(self, f'cb{res}')
            #     mask_y = cb(x_orig, mask_y) # mask_x 通道为 mid_channels

            # if res in [32, 128]:
            #     se = getattr(self, f'se{res}')
            #     x = se(x_orig, x)

            feats[res // 2] = x

        y = None
        # for idx, res in enumerate(self.dec_resolutions):
        #     block = getattr(self, f'b{res}_dec')
        #     if idx == 0:
        #         y = feats[res // 2]
        #     else:
        #         y = torch.cat([y, feats[res // 2]], dim=1)
        #
        #     y = block(y)
        #     feats[res] = y

        if get_feats:
            return img, mask_x, y
        else:
            return img, mask_x, None


@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 # w2_dim,                      # Intermediate latent (W) dimensionality.
                 num_ws,  # Number of intermediate latents to output, None = do not broadcast.
                 num_ws2,  # Number of intermediate latents to output, None = do not broadcast.
                 num_layers=8,  # Number of mapping lay       ers.
                 embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
                 layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
                 w_avg_beta=0.995,  # Decay for tracking the moving average of W during training, None = do not track.
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
        return x


# ---------------------------------------------------------------------------

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        h = self.out_layers(h)

        return self.skip_connection(x) + h

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

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4


        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = conv_nd(dims, in_channels, model_channels, 3, padding=1)

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(*layers)
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    Downsample(ch, conv_resample, dims=dims)
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = nn.Sequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch,  num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )


    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        h = h.type(x.dtype)
        return self.out(h)

@persistence.persistent_class
class Sing_ch_ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=1,  up=False, down=False):
        super().__init__()

        self.layer1 = nn.Sequential(
            normalization(in_ch),
            SiLU(),
            torch.nn.Conv2d(in_ch, out_ch, groups=groups, kernel_size=3, padding=1)
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_ch, False)
            self.x_upd = Upsample(in_ch, False)
        elif down:
            self.h_upd = Downsample(in_ch, False)
            self.x_upd = Downsample(in_ch, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.layer2 = nn.Sequential(
            normalization(out_ch),
            SiLU(),
            zero_module(torch.nn.Conv2d(out_ch, out_ch, groups=groups, kernel_size=3, padding=1))
        )

        # self.skip_connection = nn.Identity()
        self.skip_connection = torch.nn.Conv2d(in_ch, out_ch, groups=groups, kernel_size=1)

    def forward(self, x):

        if self.updown:
            in_rest, in_conv = self.layer1[:-1], self.layer1[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.layer1(x)

        h = self.layer2(h)

        return self.skip_connection(x) + h


class AdaModel(nn.Module):
    def __init__(self,
                 in_ch = 256,
                 num_head = 8,
                 dropout = 0,

                 ):
        super().__init__()
        out_ch = in_ch*2
        self.b32_16 = nn.Sequential(
            Sing_ch_ResBlock(in_ch, out_ch, groups=in_ch, down=True),
            # AttentionBlock(out_ch, num_heads=num_head),
        )

        self.b16_8 = nn.Sequential(
            Sing_ch_ResBlock(out_ch, out_ch, groups=in_ch, down=True),
            # AttentionBlock(out_ch, num_heads=num_head),
        )

        self.b8_8 = nn.Sequential(
            Sing_ch_ResBlock(out_ch, out_ch, groups=in_ch),
            # AttentionBlock(out_ch, num_heads=num_head),
            # Sing_ch_ResBlock(out_ch, out_ch, groups=in_ch),
        )

        self.b8_16 = nn.Sequential(
            Sing_ch_ResBlock(out_ch*2, out_ch, groups=in_ch, up=True),
            # AttentionBlock(out_ch, num_heads=num_head),
        )
        self.b16_32 = nn.Sequential(
            Sing_ch_ResBlock(out_ch * 2, out_ch, groups=in_ch, up=True),
            # AttentionBlock(out_ch, num_heads=num_head),
        )
        self.out = nn.Sequential(
            normalization(out_ch),
            SiLU(),
            zero_module(torch.nn.Conv2d(out_ch, in_ch, kernel_size=3, padding=1)),
        )

    def forward(self, x):
        xs = []
        x = self.b32_16(x)
        xs.append(x)
        x = self.b16_8(x)
        xs.append(x)
        x = self.b8_8(x)

        b,c,h,w = x.shape

        x = torch.cat([x.unsqueeze(2), xs.pop().unsqueeze(2)], dim=2).view(b, -1, h, w)
        x = self.b8_16(x)

        b, c, h, w = x.shape
        x = torch.cat([x.unsqueeze(2), xs.pop().unsqueeze(2)], dim=2).view(b, -1, h, w)
        x = self.b16_32(x)
        x = self.out(x)
        return x

@persistence.persistent_class
class SimGenerator(nn.Module):
    def __init__(self,
                 z_dim=512,
                 w_dim=512,
                 c_dim=512,
                 img_resolution=256,
                 img_channels=1,
                 bbox_dim=128,
                 single_size=32,
                 mid_size=64,
                 min_feat_size=8,
                 mapping_kwargs={},
                 synthesis_kwargs={}
                 ):
        super().__init__()
        assert mid_size < img_resolution
        assert min_feat_size < mid_size and mid_size % min_feat_size == 0
        self.img_resolution = img_resolution
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.bbox_dim = bbox_dim
        self.log_size = int(math.log(img_resolution, 2))
        self.img_channels = img_channels
        self.mid_size = mid_size
        self.min_feat_size = min_feat_size
        self.single_size = single_size
        self.hidden_dim = 64

        # self.embbeding_bbox = FullyConnectedLayer(4, self.hidden_dim)
        # self.embbeding_class = nn.Embedding(2, self.hidden_dim)
        # self.transform = Transformer(n_ctx=bbox_dim, width=self.hidden_dim, layers=4, heads=8)
        # self.xy_bias = FullyConnectedLayer(self.hidden_dim, 2)

        # self.ada = AdaModel()

        self.gen_single = LocalGenerator(w_dim=w_dim, img_resolution=self.single_size, img_channels=1,
                                         **synthesis_kwargs)
        self.num_ws = self.gen_single.num_ws
        self.render_net = RenderNet(w_dim=w_dim, in_resolution=min_feat_size, img_resolution=img_resolution,
                                    img_channels=img_channels, mask_channels=self.bbox_dim, mid_size=mid_size,
                                    mid_channels=bbox_dim, **synthesis_kwargs)
        self.num_ws2 = self.render_net.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_ws=self.num_ws, num_ws2=self.num_ws2,
                                      **mapping_kwargs)

    def xywh2x0y0x1y1(self, bbx):
        bbox = bbx.clone()
        bbox[:, :, 0] = bbox[:, :, 0] - bbox[:, :, 2] / 2
        bbox[:, :, 1] = bbox[:, :, 1] - bbox[:, :, 3] / 2
        bbox[:, :, 2] = bbox[:, :, 0] + bbox[:, :, 2]
        bbox[:, :, 3] = bbox[:, :, 1] + bbox[:, :, 3]
        return bbox

    def effective_bbox(self, bbox):
        bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = torch.clamp(bbox, min=0, max=1)
        # bbox[:, :, 2] = bbox[:, :, 2] - bbox[:, :, 0]
        # bbox[:, :, 3] = bbox[:, :, 3] - bbox[:, :, 1]
        return bbox

    def proj_bbox(self, bbox, label):
        sem_cond = self.embbeding_bbox(bbox.to(torch.float32))
        # sem_cond = self.transform(sem_cond)
        bias_xy = nn.Tanh()(self.xy_bias(sem_cond))
        return bias_xy[:, :, 0], bias_xy[:, :, 1]

    def bias_sin_ms(self, sin_ms, bbx, labels):
        bbox = self.xywh2x0y0x1y1(bbx)
        bias_x, bias_y = self.proj_bbox(bbox, labels)



        bias_x, bias_y = bias_x.unsqueeze(-1), bias_y.unsqueeze(-1)
        # bias = torch.cat([bias_x, bias_y, bias_x, bias_y], dim=-1)
        theta_x = torch.cat([torch.ones_like(bias_x), torch.zeros_like(bias_x), bias_x], dim=-1)
        theta_y = torch.cat([torch.zeros_like(bias_y), torch.ones_like(bias_y), bias_y], dim=-1)
        theta = torch.cat([theta_x.unsqueeze(2), theta_y.unsqueeze(2)], dim=2).view(-1, 2, 3)
        minus_bias_x_div_2, minus_bias_y_div_2 = -bias_x / 2, -bias_y / 2
        theta2_x = torch.cat([torch.ones_like(minus_bias_x_div_2) * (1 - torch.abs(minus_bias_x_div_2)),
                              torch.zeros_like(minus_bias_x_div_2), minus_bias_x_div_2], dim=-1)
        theta2_y = torch.cat([torch.zeros_like(minus_bias_y_div_2),
                              torch.ones_like(minus_bias_y_div_2) * (1 - torch.abs(minus_bias_y_div_2)),
                              minus_bias_y_div_2], dim=-1)
        theta2 = torch.cat([theta2_x.unsqueeze(2), theta2_y.unsqueeze(2)], dim=2).view(-1, 2, 3)

        sin_grid = F.affine_grid(theta=theta, size=sin_ms.size(), align_corners=True)
        sin_ms = grid_sample_gradfix.grid_sample(sin_ms, sin_grid)
        sin_grid = F.affine_grid(theta=theta2, size=sin_ms.size(), align_corners=True)
        sin_ms = grid_sample_gradfix.grid_sample(sin_ms, sin_grid)
        return sin_ms

    def forward(self, z, bbox, truncation_psi=1, truncation_cutoff=None, isTrain=True, get_feat=True):
        misc.assert_shape(bbox, [None, self.bbox_dim, 5])
        labels, bbox = bbox[:, :, 0], bbox[:, :, 1:]
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        sin_m = (self.gen_single(ws[:, :self.num_ws]) + 1.0) / 2.0  # adjust from [-1, 1] to [0, 1]
        sin_ms = sin_m.repeat(1, self.bbox_dim, 1, 1).view(-1, 1, self.single_size, self.single_size)

        # sin_ms = self.bias_sin_ms(sin_ms, bbox, labels)

        grid = _boxes_to_grid(bbox.view(-1, 4), self.mid_size, self.mid_size).to(sin_ms.dtype)
        mid_masks = grid_sample_gradfix.grid_sample(sin_ms, grid).view(-1, self.bbox_dim, self.mid_size, self.mid_size)
        mid_masks = mid_masks * bbox_mask(z.device, bbox, self.mid_size, self.mid_size)
        # mid_masks = mid_masks.mul(2.0) - 1.0 # [0, 1]adjust to [-1, 1]

        # mid_masks = self.ada(mid_masks)

        img, sem_mask, feats = self.render_net(mid_masks*2.0-1.0, ws[:, self.num_ws:], bbox, get_feats=False)  # mask[0, 1]
        # print(feats.shape)
        # print(sem_mask.shape, mid_masks.shape)
        # for val in feats:
        #     print(val,":", feats[val].shape)

        if isTrain:
            # print(sem_mask.shape)
            # sem_mask = self.numeric_sem_mask(sem_mask)
            # mid_masks = self.numeric_sem_mask(mid_masks)
            return img, sem_mask, feats, mid_masks.sum(dim=1, keepdim=True).clamp(0, 1) * 2 - 1
            # print(mid_masks)
            # return img, sem_mask, ws, mid_masks
            # return img, torch.max(mask*0.5+0.5, dim=1).values.unsqueeze(1), ws, torch.max(mid_masks, dim=1).values.unsqueeze(1)
        else:
            return img, mid_masks, sem_mask

    def numeric_sem_mask(self, sem_mask):
        sem_mask = sem_mask.clamp(-1, 1) * 0.5 + 0.5  # [-1, 1] to [0, 1]
        B, ch = sem_mask.shape[:2]
        weight = torch.arange(ch, 0, -1).to(sem_mask.device).unsqueeze(0).repeat([B, 1]).view(B, ch, 1, 1)
        wmask = sem_mask * weight
        wmask = torch.max(wmask, dim=1).values.unsqueeze(1)
        return wmask

# ----------------------------------------------------------------------------
@persistence.persistent_class
class DualDiscriminator(torch.nn.Module):
    def __init__(self,
                 img_resolution=256,  # Input resolution.
                 img_channels=3,  # Number of input color channels.
                 mask_channels=1,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 single=False,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 block_kwargs={},  # Arguments for DiscriminatorBlock.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
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

            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       img_channels=img_channels,
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs,
                                       **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        self.ob32 = DiscriminatorBlock(channels_dict[32], channels_dict[32], channels_dict[16], resolution=32,
                                       img_channels=img_channels, down=False,
                                       first_layer_idx=cur_layer_idx, use_fp16=(32 >= fp16_resolution), **block_kwargs,
                                       **common_kwargs)
        cur_layer_idx += self.ob32.num_layers

        self.ob16 = DiscriminatorBlock(channels_dict[32], channels_dict[32], channels_dict[16], resolution=16,
                                       img_channels=img_channels, down=False,
                                       first_layer_idx=cur_layer_idx, use_fp16=(16 >= fp16_resolution), **block_kwargs,
                                       **common_kwargs)
        cur_layer_idx += self.ob16.num_layers

        self.ob_last = DiscriminatorBlock(channels_dict[8], channels_dict[8], channels_dict[4],
                                          first_layer_idx=cur_layer_idx, use_fp16=(8 >= fp16_resolution), resolution=8,
                                          img_channels=img_channels, **block_kwargs)
        self.roi_align_s = ROIAlign((8, 8), 1.0 / 4.0, int(0))
        self.roi_align_l = ROIAlign((8, 8), 1.0 / 8.0, int(0))
        self.b4 = DiscriminatorEpilogue(channels_dict[4], resolution=4, getVec=False, **epilogue_kwargs,
                                        **common_kwargs)
        self.ob4 = DiscriminatorEpilogue(channels_dict[4], resolution=4, getVec=False, **epilogue_kwargs,
                                         **common_kwargs)

    def xywh2x0y0x1y1(self, bbox1):
        bbox = bbox1.clone()
        bbox[:, :, 1] = bbox1[:, :, 1] - bbox1[:, :, 3] / 2
        bbox[:, :, 2] = bbox1[:, :, 2] - bbox1[:, :, 4] / 2
        bbox[:, :, 3] = bbox1[:, :, 1] + bbox1[:, :, 3]
        bbox[:, :, 4] = bbox1[:, :, 2] + bbox1[:, :, 4]
        return bbox

    def pre_bbox(self, bbox):
        bbox[:, :, 1:] = (bbox[:, :, 1:] * (self.img_resolution - 1)).clamp(0, self.img_resolution - 1).to(torch.uint8)
        label, bbox = bbox[:, :, 0], bbox[:, :, 1:]
        idx = torch.arange(start=0, end=bbox.size(0), device=bbox.device).view(bbox.size(0), 1, 1).expand(-1,
                                                                                                          bbox.size(1),
                                                                                                          -1).float()
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        return label, bbox

    def classifier(self, label, bbox):
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 64) * ((bbox[:, 4] - bbox[:, 2]) < 64)
        bbox_s, bbox_l = bbox[s_idx], bbox[~s_idx]
        label_s, label_l = label[s_idx], label[~s_idx]
        return bbox_s, bbox_l

    def forward(self, img, bbox, mask, **block_kwargs):
        bbox = self.xywh2x0y0x1y1(bbox)
        pre_label, pre_bbox = self.pre_bbox(bbox)
        bbox_s, bbox_l = self.classifier(pre_label, pre_bbox)

        x = None
        x1 = x2 = None

        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

            if res == 64:
                x1 = x
            if res == 32:
                x2 = x
        x1, _ = self.ob32(x1, img, **block_kwargs)
        x1, _ = self.ob16(x1, img, **block_kwargs)
        x1 = self.roi_align_s(x1, bbox_s)

        x2, _ = self.ob16(x2, img, **block_kwargs)
        x2 = self.roi_align_l(x2, bbox_l)

        obj_feat, _ = self.ob_last(torch.cat([x1, x2], dim=0), img, **block_kwargs)
        x = self.b4(x)
        N, num_o = x.shape[0], obj_feat.shape[0]
        groups = num_o // N
        y = self.ob4(obj_feat[:groups * N]).view(N, -1, 1).mean(dim=1)
        return x, y


@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
                 img_resolution=256,  # Input resolution.
                 img_channels=3,  # Number of input color channels.
                 mask_channels=1,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 block_kwargs={},  # Arguments for DiscriminatorBlock.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
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

    def forward(self, img, bbox, mask, **block_kwargs):
        d1 = self.Mnet(img, bbox, mask, **block_kwargs)
        # samples = img_resampler(img, bbox, resample_num=16) # (B*16, C, H, W)
        # d2 = self.Snet(samples, bbox, mask, **block_kwargs).view(img.shape[0], 16, 1).mean(1)
        # return d1, d2
        return d1


if __name__ == "__main__":
    net = AdaModel()
    x= torch.rand([4, 256, 32, 32])
    y = net(x)
    print(y.shape)