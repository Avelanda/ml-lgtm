"""
DepthSplat backbone.

For third-party code see ACKNOWLEDGMENTS file.

By using the DepthSplat backbone, you agree to comply with the licenses of
DepthSplat and all associated dependencies, including but not limited to:
- depthsplat: https://github.com/cvg/depthsplat/blob/main/LICENSE
- unimatch: https://github.com/autonomousvision/unimatch/blob/master/LICENSE
- dinov2: https://github.com/facebookresearch/dinov2/blob/main/LICENSE
"""

import math
import os
from abc import abstractmethod
from inspect import isfunction

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [
            x.detach().requires_grad_(True) for x in ctx.input_tensors
        ]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels, channels_per_group=None):
    """
    Make a standard normalization layer.
    """
    if channels_per_group is not None:
        return GroupNorm(channels // channels_per_group, channels)

    if channels % 8 != 0:
        return GroupNorm4(4, channels)

    return GroupNorm8(8, channels)


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class GroupNorm8(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class GroupNorm4(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def _exists(val):
    return val is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if isfunction(d) else d


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = _default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def _Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class _LdmCrossAttention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = _default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = _default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v)
        )

        from torch import einsum

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if _exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=False,
    ):
        super().__init__()
        self.attn1 = _LdmCrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = _LdmCrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        # self.checkpoint = checkpoint

    def forward(self, x, context=None):
        # return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

        return self._forward(x, context)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = _Normalize(in_channels)

        self.proj_in = nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                )
                for d in range(depth)
            ]
        )

        self.proj_out = zero_module(
            nn.Conv2d(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0
            )
        )

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class _XFormersCrossAttention(nn.Module):
    def __init__(
        self,
        in_dim1,
        in_dim2,
        dim=128,
        out_dim=None,
        num_heads=4,
        qkv_bias=False,
        proj_bias=False,
    ):
        super().__init__()

        if not XFORMERS_AVAILABLE:
            raise RuntimeError("xformers is required but not available")

        if out_dim is None:
            out_dim = in_dim1

        self.num_heads = num_heads
        self.dim = dim
        self.q = nn.Linear(in_dim1, dim, bias=qkv_bias)
        self.kv = nn.Linear(in_dim2, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, out_dim, bias=proj_bias)

    def forward(self, x, y):
        c = self.dim
        b, n1, c1 = x.shape
        n2, c2 = y.shape[1:]

        q = self.q(x).reshape(b, n1, self.num_heads, c // self.num_heads)
        kv = self.kv(y).reshape(b, n2, 2, self.num_heads, c // self.num_heads)
        k, v = unbind(kv, 2)

        x = memory_efficient_attention(q, k, v)
        x = x.reshape(b, n1, c)

        x = self.proj(x)

        return x


class UNetCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        in_dim1,
        in_dim2,
        dim=128,
        out_dim=None,
        num_heads=4,
        qkv_bias=False,
        proj_bias=False,
        with_ffn=False,
        concat_cross_attn=False,
        concat_output=False,
        no_cross_attn=False,
        with_norm=False,
        concat_conv3x3=False,
    ):
        super().__init__()

        out_dim = out_dim or in_dim1

        self.no_cross_attn = no_cross_attn
        self.with_norm = with_norm

        if no_cross_attn:
            if concat_conv3x3:
                self.proj = nn.Conv2d(in_dim1 + in_dim2, out_dim, 3, 1, 1)
            else:
                self.proj = nn.Conv2d(in_dim1 + in_dim2, out_dim, 1)
        else:
            self.with_ffn = with_ffn
            self.concat_cross_attn = concat_cross_attn
            self.concat_output = concat_output

            self.cross_attn = _XFormersCrossAttention(
                in_dim1=in_dim1,
                in_dim2=in_dim2,
                dim=dim,
                out_dim=out_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
            )

            if with_norm:
                self.norm1 = nn.LayerNorm(out_dim)
            else:
                self.norm1 = nn.Identity()

            if with_ffn:
                in_channels = (
                    out_dim + in_dim1 if concat_cross_attn else in_dim1
                )
                ffn_dim_expansion = 4
                self.mlp = nn.Sequential(
                    nn.Linear(
                        in_channels, in_channels * ffn_dim_expansion, bias=False
                    ),
                    nn.GELU(),
                    nn.Linear(
                        in_channels * ffn_dim_expansion, in_dim1, bias=False
                    ),
                )

                if with_norm:
                    self.norm2 = nn.LayerNorm(in_dim1)
                else:
                    self.norm2 = nn.Identity()

            if self.concat_output:
                self.out = nn.Linear(out_dim + in_dim1, in_dim1)

    def forward(self, x, y):
        if self.no_cross_attn:
            if not (x.dim() == 4 and y.dim() == 4):
                raise ValueError("x and y must both be 4-dimensional tensors")
            if y.shape[2:] != x.shape[2:]:
                y = F.interpolate(
                    y, x.shape[2:], mode="bilinear", align_corners=True
                )
            return self.proj(torch.cat((x, y), dim=1))

        identity = x

        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)

        cross_attn = self.norm1(self.cross_attn(x, y))

        if self.with_ffn:
            if self.concat_cross_attn:
                concat = torch.cat((x, cross_attn), dim=-1)
            else:
                concat = x + cross_attn

            cross_attn = self.norm2(self.mlp(concat))

        if self.concat_output:
            return self.out(torch.cat((x, cross_attn), dim=-1))

        cross_attn = cross_attn.view(b, h, w, c).permute(
            0, 3, 1, 2
        )  # [B, C, H, W]

        return identity + cross_attn


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels,
        use_conv,
        dims=2,
        out_channels=None,
        padding=1,
        downsample_3ddim=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=padding
            )

        self.downsample_3ddim = downsample_3ddim

    def forward(self, x, y=None):
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected x.shape[1] == {self.channels}, got {x.shape[1]}"
            )
        if self.dims == 3 and not self.downsample_3ddim:
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
    """

    def __init__(
        self,
        channels,
        use_conv,
        dims=2,
        out_channels=None,
        padding=1,
        downsample_3ddim=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)

        if downsample_3ddim:
            if dims != 3:
                raise ValueError("downsample_3ddim requires dims == 3")
            stride = 2

        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            if self.channels != self.out_channels:
                raise ValueError(
                    "channels must equal out_channels when use_conv is False"
                )
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x, y=None):
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected x.shape[1] == {self.channels}, got {x.shape[1]}"
            )
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
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
        postnorm=False,
        channels_per_group=None,
        kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        if postnorm:
            self.in_layers = nn.Sequential(
                conv_nd(
                    dims,
                    channels,
                    self.out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                normalization(
                    self.out_channels, channels_per_group=channels_per_group
                ),
                nn.SiLU(),
            )
        else:
            self.in_layers = nn.Sequential(
                normalization(channels, channels_per_group=channels_per_group),
                nn.SiLU(),
                conv_nd(
                    dims,
                    channels,
                    self.out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
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

        if postnorm:
            self.out_layers = nn.Sequential(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                zero_module(
                    normalization(
                        self.out_channels, channels_per_group=channels_per_group
                    ),
                ),
                nn.SiLU(),
            )
        else:
            self.out_layers = nn.Sequential(
                normalization(
                    self.out_channels, channels_per_group=channels_per_group
                ),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(
                        dims,
                        self.out_channels,
                        self.out_channels,
                        kernel_size,
                        padding=(kernel_size - 1) // 2,
                    )
                ),
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims,
                channels,
                self.out_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb=None):
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


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        postnorm=False,
        channels_per_group=None,
        num_frames=2,
        use_cross_view_self_attn=False,
    ):
        super().__init__()

        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            if channels % num_head_channels != 0:
                raise ValueError(
                    f"q,k,v channels {channels} is not divisible by"
                    f" num_head_channels {num_head_channels}"
                )
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(
                self.num_heads,
                n_frames=num_frames,
                use_cross_view_self_attn=use_cross_view_self_attn,
            )

        if postnorm:
            self.proj_out = conv_nd(1, channels, channels, 1)
            self.norm = zero_module(
                normalization(channels, channels_per_group=channels_per_group)
            )
        else:
            self.norm = normalization(
                channels, channels_per_group=channels_per_group
            )
            self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

        self.postnorm = postnorm

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        if self.postnorm:
            qkv = self.qkv(x)
            h = self.attention(qkv)
            h = self.proj_out(h)
            h = self.norm(h)
        else:
            qkv = self.qkv(self.norm(x))
            h = self.attention(qkv)
            h = self.proj_out(h)

        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads, n_frames=2, use_cross_view_self_attn=False):
        super().__init__()
        self.n_heads = n_heads
        self.n_frames = n_frames
        self.use_cross_view_self_attn = use_cross_view_self_attn

    def forward(self, qkv, num_views=None):
        if self.use_cross_view_self_attn:
            n_views = self.n_frames if num_views is None else num_views
            qkv = rearrange(qkv, "(b v) n t -> b n (v t)", v=n_views)

        bs, width, length = qkv.shape
        if width % (3 * self.n_heads) != 0:
            raise ValueError(
                f"width {width} must be divisible by 3 * n_heads"
                f" {3 * self.n_heads}"
            )
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(
            ch, dim=1
        )
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v).reshape(bs, -1, length)

        if self.use_cross_view_self_attn:
            a = rearrange(a, "b n (v t) -> (b v) n t", v=n_views)

        return a

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        if width % (3 * self.n_heads) != 0:
            raise ValueError(
                f"width {width} must be divisible by 3 * n_heads"
                f" {3 * self.n_heads}"
            )
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        middle_block_attn=False,
        middle_block_no_identity=False,
        postnorm=False,
        attn_prenorm=False,
        downsample_3ddim=False,
        zero_final_layer=False,
        channels_per_group=None,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        cross_attn_condition=False,
        tanh_gating=False,
        ffn_after_cross_attn=False,
        cross_attn_with_norm=False,
        condition_channels=384,
        condition_num_views=3,
        no_self_attn=False,
        conv_kernel_size=3,
        concat_condition=False,
        concat_conv3x3=False,
        num_frames=2,
        use_cross_view_self_attn=False,
        downsample_factor=None,
    ):
        super().__init__()
        if use_spatial_transformer:
            if context_dim is None:
                raise ValueError(
                    "Fool!! You forgot to include the dimension of your"
                    " cross-attention conditioning..."
                )

        if context_dim is not None:
            if not use_spatial_transformer:
                raise ValueError(
                    "Fool!! You forgot to use the spatial transformer for your"
                    " cross-attention conditioning..."
                )
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            if num_head_channels == -1:
                raise ValueError(
                    "Either num_heads or num_head_channels has to be set"
                )

        if num_head_channels == -1:
            if num_heads == -1:
                raise ValueError(
                    "Either num_heads or num_head_channels has to be set"
                )

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        self.middle_block_attn = middle_block_attn

        self.middle_block_no_identity = middle_block_no_identity

        self.downsample_factor = downsample_factor

        time_embed_dim = model_channels * 4

        self.cross_attn_condition = cross_attn_condition

        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
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
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )

                    if not no_self_attn:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                                postnorm=False if attn_prenorm else postnorm,
                                channels_per_group=channels_per_group,
                                num_frames=num_frames,
                                use_cross_view_self_attn=use_cross_view_self_attn,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                            )
                        )

                    if cross_attn_condition:
                        layers.append(
                            UNetCrossAttentionBlock(
                                ch,
                                condition_channels,
                                dim=256,
                                no_cross_attn=concat_condition,
                                with_norm=cross_attn_with_norm,
                                concat_conv3x3=concat_conv3x3,
                            )
                        )

                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            postnorm=postnorm,
                            channels_per_group=channels_per_group,
                            kernel_size=conv_kernel_size,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            downsample_3ddim=downsample_3ddim,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = (
                ch // num_heads
                if use_spatial_transformer
                else num_head_channels
            )

        if self.middle_block_attn:
            self.middle_block = nn.Sequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    postnorm=postnorm,
                    channels_per_group=channels_per_group,
                    kernel_size=conv_kernel_size,
                ),
                (
                    AttentionBlock(
                        ch,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads,
                        num_head_channels=dim_head,
                        use_new_attention_order=use_new_attention_order,
                        postnorm=False if attn_prenorm else postnorm,
                        channels_per_group=channels_per_group,
                        num_frames=num_frames,
                        use_cross_view_self_attn=use_cross_view_self_attn,
                    )
                    if not use_spatial_transformer
                    else SpatialTransformer(
                        ch,
                        num_heads,
                        dim_head,
                        depth=transformer_depth,
                        context_dim=context_dim,
                    )
                ),
                (
                    UNetCrossAttentionBlock(
                        ch,
                        condition_channels,
                        dim=256,
                        no_cross_attn=concat_condition,
                        with_norm=cross_attn_with_norm,
                        concat_conv3x3=concat_conv3x3,
                    )
                    if cross_attn_condition
                    else nn.Identity()
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    postnorm=postnorm,
                    channels_per_group=channels_per_group,
                    kernel_size=conv_kernel_size,
                ),
            )
        else:
            if self.middle_block_no_identity:
                self.middle_block = nn.Sequential(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    ),
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    ),
                )
            else:
                self.middle_block = nn.Sequential(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    ),
                    (
                        UNetCrossAttentionBlock(
                            ch,
                            condition_channels,
                            dim=256,
                            no_cross_attn=concat_condition,
                            with_norm=cross_attn_with_norm,
                            concat_conv3x3=concat_conv3x3,
                        )
                        if cross_attn_condition
                        else nn.Identity()
                    ),
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    ),
                )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )

                    if not no_self_attn:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                                postnorm=False if attn_prenorm else postnorm,
                                channels_per_group=channels_per_group,
                                num_frames=num_frames,
                                use_cross_view_self_attn=use_cross_view_self_attn,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                            )
                        )

                    if cross_attn_condition:
                        layers.append(
                            UNetCrossAttentionBlock(
                                ch,
                                condition_channels,
                                dim=256,
                                no_cross_attn=concat_condition,
                                with_norm=cross_attn_with_norm,
                                concat_conv3x3=concat_conv3x3,
                            )
                        )

                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            postnorm=postnorm,
                            channels_per_group=channels_per_group,
                            kernel_size=conv_kernel_size,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            downsample_3ddim=downsample_3ddim,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        if self.downsample_factor is not None:
            if self.downsample_factor == 2:
                del self.output_blocks[-3:]
            elif self.downsample_factor == 4:
                del self.output_blocks[-5:]
            else:
                raise NotImplementedError

        if postnorm:
            self.out = nn.Sequential(
                conv_nd(dims, model_channels, out_channels, 3, padding=1),
                (
                    normalization(
                        out_channels, channels_per_group=channels_per_group
                    )
                    if not zero_final_layer
                    else zero_module(
                        normalization(
                            out_channels, channels_per_group=channels_per_group
                        )
                    )
                ),
                nn.SiLU(),
            )
        else:
            if self.downsample_factor is not None:
                in_channels = (
                    self.model_channels
                    * self.channel_mult[self.downsample_factor // 2]
                )
                model_channels = (
                    model_channels
                    * self.channel_mult[self.downsample_factor // 2]
                )
                out_channels = model_channels
            else:
                in_channels = ch
            self.out = nn.Sequential(
                normalization(
                    in_channels, channels_per_group=channels_per_group
                ),
                nn.SiLU(),
                zero_module(
                    conv_nd(dims, model_channels, out_channels, 3, padding=1)
                ),
            )

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch, channels_per_group=channels_per_group),
                conv_nd(dims, model_channels, n_embed, 1),
            )

    def forward(
        self, x, num_views=None, timesteps=None, context=None, y=None, **kwargs
    ):
        if (y is not None) != (self.num_classes is not None):
            raise ValueError(
                "must specify y if and only if the model is class-conditional"
            )
        hs = []
        emb = None

        if self.num_classes is not None:
            if y.shape != (x.shape[0],):
                raise ValueError(
                    f"y.shape must be ({x.shape[0]},), got {y.shape}"
                )
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            if self.cross_attn_condition:
                for submodule in module:
                    if (
                        "UNetCrossAttentionBlock"
                        == submodule.__class__.__name__
                    ):
                        h = submodule(h, context)
                    else:
                        h = submodule(h)
            else:
                h = module(h)
            hs.append(h)

        for module in self.middle_block:
            if "UNetCrossAttentionBlock" == module.__class__.__name__:
                h = module(h, context)
            else:
                h = module(h)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            if self.cross_attn_condition:
                for submodule in module:
                    if (
                        "UNetCrossAttentionBlock"
                        == submodule.__class__.__name__
                    ):
                        h = submodule(h, context)
                    else:
                        h = submodule(h)
            else:
                h = module(h)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the
    one used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=True, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=x.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        norm_layer=nn.InstanceNorm2d,
        stride=1,
        dilation=1,
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            stride=stride,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        output_dim=128,
        norm_layer=nn.InstanceNorm2d,
        num_output_scales=1,
        return_quarter=False,
        lowest_scale=8,
        return_all_scales=False,
        **kwargs,
    ):
        super(CNNEncoder, self).__init__()
        self.num_scales = num_output_scales
        self.return_quarter = return_quarter
        self.lowest_scale = lowest_scale
        self.return_all_scales = return_all_scales

        feature_dims = [64, 96, 128]

        self.conv1 = nn.Conv2d(
            3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(
            feature_dims[0], stride=1, norm_layer=norm_layer
        )

        if self.lowest_scale == 4:
            stride = 1
        else:
            stride = 2
        self.layer2 = self._make_layer(
            feature_dims[1], stride=stride, norm_layer=norm_layer
        )

        self.layer3 = self._make_layer(
            feature_dims[2],
            stride=2,
            norm_layer=norm_layer,
        )

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(
                m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)
            ):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d
    ):
        layer1 = ResidualBlock(
            self.in_planes,
            dim,
            norm_layer=norm_layer,
            stride=stride,
            dilation=dilation,
        )
        layer2 = ResidualBlock(
            dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation
        )

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        output_all_scales = []
        output = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)

        if self.return_all_scales:
            output_all_scales.append(x)

        if self.num_scales >= 3:
            output.append(x)

        x = self.layer2(x)
        if self.return_quarter:
            output.append(x)

        if self.return_all_scales:
            output_all_scales.append(x)

        if self.num_scales >= 2:
            output.append(x)

        x = self.layer3(x)
        x = self.conv2(x)

        if self.return_all_scales:
            output_all_scales.append(x)

        if self.return_all_scales:
            return output_all_scales

        if self.return_quarter:
            output.append(x)
            return output

        if self.num_scales >= 1:
            output.append(x)
            return output

        out = [x]

        return out


def split_feature(
    feature,
    num_splits=2,
    channel_last=False,
):
    if channel_last:
        b, h, w, c = feature.size()
        if not (h % num_splits == 0 and w % num_splits == 0):
            raise ValueError(
                f"h={h} and w={w} must be divisible by num_splits={num_splits}"
            )

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(
                b, num_splits, h // num_splits, num_splits, w // num_splits, c
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b_new, h_new, w_new, c)
        )
    else:
        b, c, h, w = feature.size()
        if not (h % num_splits == 0 and w % num_splits == 0):
            raise ValueError(
                f"h={h} and w={w} must be divisible by num_splits={num_splits}"
            )

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(
                b, c, num_splits, h // num_splits, num_splits, w // num_splits
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b_new, c, h_new, w_new)
        )

    return feature


def merge_splits(
    splits,
    num_splits=2,
    channel_last=False,
):
    if channel_last:
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = (
            splits.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(new_b, num_splits * h, num_splits * w, c)
        )
    else:
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = (
            splits.permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(new_b, c, num_splits * h, num_splits * w)
        )

    return merge


def generate_shift_window_attn_mask(
    input_resolution,
    window_size_h,
    window_size_w,
    shift_size_h,
    shift_size_w,
    device=torch.device("cuda"),
):
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(
        img_mask,
        num_splits=input_resolution[-1] // window_size_w,
        channel_last=True,
    )

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(
        attn_mask != 0, float(-100.0)
    ).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def mv_feature_add_position(features, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if features.dim() != 4:
        raise ValueError("features must be 4-dimensional [B*V, C, H, W]")

    if attn_splits > 1:
        features_splits = split_feature(features, num_splits=attn_splits)
        position = pos_enc(features_splits)
        features_splits = features_splits + position
        features = merge_splits(features_splits, num_splits=attn_splits)
    else:
        position = pos_enc(features)
        features = features + position

    return features


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()

    grid = grid[None].repeat(b, 1, 1, 1)

    if device is not None:
        grid = grid.to(device)

    return grid


def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    grid_sample_disable_cudnn=False,
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W].
    """

    if not (intrinsics.size(1) == intrinsics.size(2) == 3):
        raise ValueError("intrinsics must be a 3x3 matrix")
    if not (pose.size(1) == pose.size(2) == 4):
        raise ValueError("pose must be a 4x4 matrix")
    if depth.dim() != 4:
        raise ValueError("depth must be 4-dimensional")

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        grid = coords_grid(b, h, w, homogeneous=True, device=depth.device)
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(b, 1, d, h * w)
        points = points + pose[:, :3, -1:].unsqueeze(-1)
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )
        pixel_coords = points[:, :2] / points[:, -1:].clamp(min=clamp_min_depth)

        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)

    if feature1.numel() > 1000000:
        grid_sample_disable_cudnn = True
    with torch.backends.cudnn.flags(enabled=not grid_sample_disable_cudnn):
        warped_feature = F.grid_sample(
            feature1,
            grid.view(b, d * h, w, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(b, c, d, h, w)

    return warped_feature


class ViTFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    """

    def __init__(
        self,
        in_channels,
        scale_factors,
    ):
        super(ViTFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        out_dim = dim = in_channels
        self.stages = nn.ModuleList()
        for idx, scale in enumerate(scale_factors):
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        dim // 2, dim // 4, kernel_size=2, stride=2
                    ),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
                ]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(
                    f"scale_factor={scale} is not supported yet."
                )

            if scale != 1.0:
                layers.extend(
                    [
                        nn.GELU(),
                        nn.Conv2d(out_dim, out_dim, 3, 1, 1),
                    ]
                )
            layers = nn.Sequential(*layers)

            self.stages.append(layers)

    def forward(self, x):
        results = []

        for stage in self.stages:
            results.append(stage(x))

        return results


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
    ):
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output,
            **modifier,
            mode="bilinear",
            align_corners=self.align_corners,
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        concat_cnn_features=True,
        concat_mv_features=True,
        cnn_feature_channels=[64, 96, 128],
        concat_features=True,
        downsample_factor=8,
        return_feature=False,
        num_scales=1,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.concat_cnn_features = concat_cnn_features
        self.concat_mv_features = concat_mv_features
        self.concat_features = concat_features
        self.downsample_factor = downsample_factor
        self.return_feature = return_feature
        self.num_scales = num_scales

        if self.concat_features:
            if self.downsample_factor == 4 and num_scales == 2:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0] + out_channels[0],
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[1]
                            + out_channels[1]
                            + 64
                            + depth_channel,
                            out_channels[1],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[2] + out_channels[2] + 128,
                            out_channels[2],
                            1,
                        ),
                    ]
                )
            elif self.downsample_factor == 2 and num_scales == 2:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0]
                            + cnn_feature_channels[1]
                            + out_channels[0]
                            + 64
                            + depth_channel,
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[2] + out_channels[1] + 128,
                            out_channels[1],
                            1,
                        ),
                        nn.Conv2d(out_channels[2], out_channels[2], 1),
                    ]
                )
            elif self.downsample_factor == 4 and num_scales == 1:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0]
                            + cnn_feature_channels[1]
                            + out_channels[0],
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[2]
                            + out_channels[1]
                            + 128
                            + depth_channel,
                            out_channels[1],
                            1,
                        ),
                        nn.Conv2d(out_channels[2], out_channels[2], 1),
                    ]
                )
            else:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0] + out_channels[0],
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[1] + out_channels[1],
                            out_channels[1],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[2]
                            + out_channels[2]
                            + 128
                            + depth_channel,
                            out_channels[2],
                            1,
                        ),
                    ]
                )
        else:
            if self.concat_cnn_features:
                self.cnn_projects = nn.ModuleList(
                    [
                        nn.Conv2d(cnn_feature_channels[i], out_channels[i], 1)
                        for i in range(len(cnn_feature_channels))
                    ]
                )

            if self.concat_mv_features:
                self.mv_projects = nn.Conv2d(128, out_channels[2], 1)

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels), nn.GELU()
                    )
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # Not used.
        del self.scratch.refinenet4.resConfUnit1

        head_features_1 = features
        head_features_2 = 16

        if not self.return_feature:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(
                    head_features_1,
                    head_features_1 // 2,
                    3,
                    1,
                    1,
                    padding_mode="replicate",
                ),
                nn.GELU(),
                nn.Conv2d(
                    head_features_1 // 2,
                    head_features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.GELU(),
                nn.Conv2d(
                    head_features_2, 1, kernel_size=1, stride=1, padding=0
                ),
            )

            nn.init.zeros_(self.scratch.output_conv[-1].weight)
            nn.init.zeros_(self.scratch.output_conv[-1].bias)

    def forward(
        self,
        out_features,
        downsample_factor=8,
        cnn_features=None,
        mv_features=None,
        depth=None,
    ):
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        if self.concat_features:
            if not self.return_feature:
                if depth is None:
                    raise ValueError("depth must not be None")

            if self.downsample_factor == 4 and self.num_scales == 1:
                concat1 = torch.cat(
                    (cnn_features[0], cnn_features[1], layer_1), dim=1
                )
            elif self.downsample_factor == 2 and self.num_scales == 2:
                if self.return_feature:
                    concat1 = torch.cat(
                        (
                            cnn_features[0],
                            cnn_features[1],
                            mv_features[0],
                            layer_1,
                        ),
                        dim=1,
                    )
                else:
                    concat1 = torch.cat(
                        (
                            cnn_features[0],
                            cnn_features[1],
                            mv_features[0],
                            depth,
                            layer_1,
                        ),
                        dim=1,
                    )
            else:
                concat1 = torch.cat((cnn_features[0], layer_1), dim=1)
            layer_1 = self.concat_projects[0](concat1)

            if self.downsample_factor == 4 and self.num_scales == 2:
                if not isinstance(mv_features, list):
                    raise TypeError("mv_features must be a list")
                if self.return_feature:
                    concat2 = torch.cat(
                        (cnn_features[1], layer_2, mv_features[0]), dim=1
                    )
                else:
                    concat2 = torch.cat(
                        (cnn_features[1], layer_2, mv_features[0], depth), dim=1
                    )
                layer_2 = self.concat_projects[1](concat2)

                concat3 = torch.cat(
                    (cnn_features[2], layer_3, mv_features[1]), dim=1
                )
                layer_3 = self.concat_projects[2](concat3)
            elif self.downsample_factor == 2 and self.num_scales == 2:
                if not isinstance(mv_features, list):
                    raise TypeError("mv_features must be a list")
                concat2 = torch.cat(
                    (cnn_features[2], layer_2, mv_features[1]), dim=1
                )
                layer_2 = self.concat_projects[1](concat2)

                concat3 = layer_3
                layer_3 = self.concat_projects[2](concat3)
            elif self.downsample_factor == 4 and self.num_scales == 1:
                if self.return_feature:
                    concat2 = torch.cat(
                        (cnn_features[2], layer_2, mv_features), dim=1
                    )
                else:
                    concat2 = torch.cat(
                        (cnn_features[2], layer_2, mv_features, depth), dim=1
                    )
                layer_2 = self.concat_projects[1](concat2)

                concat3 = layer_3
                layer_3 = self.concat_projects[2](concat3)
            else:
                concat2 = torch.cat((cnn_features[1], layer_2), dim=1)
                layer_2 = self.concat_projects[1](concat2)

                if self.return_feature:
                    concat3 = torch.cat(
                        (cnn_features[2], layer_3, mv_features), dim=1
                    )
                else:
                    concat3 = torch.cat(
                        (cnn_features[2], layer_3, mv_features, depth), dim=1
                    )
                layer_3 = self.concat_projects[2](concat3)
        else:
            if self.concat_cnn_features:
                if cnn_features is None:
                    raise ValueError("cnn_features must not be None")
                if len(cnn_features) != 3:
                    raise ValueError(
                        "cnn_features must have exactly 3 elements"
                    )
                cnn_features = [
                    self.cnn_projects[i](f) for i, f in enumerate(cnn_features)
                ]

                layer_1 = layer_1 + cnn_features[0]
                layer_2 = layer_2 + cnn_features[1]
                layer_3 = layer_3 + cnn_features[2]

            if self.concat_mv_features:
                mv_features = self.mv_projects(mv_features)

                layer_3 = layer_3 + mv_features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(
            path_4, layer_3_rn, size=layer_2_rn.shape[2:]
        )
        path_2 = self.scratch.refinenet2(
            path_3, layer_2_rn, size=layer_1_rn.shape[2:]
        )
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.return_feature:
            return path_1

        out = self.scratch.output_conv(path_1)

        return out


def single_head_full_attention(q, k, v):
    if not (q.dim() == k.dim() == v.dim() == 3):
        raise ValueError("q, k, v must all be 3-dimensional tensors")

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** 0.5)
    attn = torch.softmax(scores, dim=2)
    out = torch.matmul(attn, v)

    return out


def single_head_split_window_attention(
    q,
    k,
    v,
    num_splits=1,
    with_shift=False,
    h=None,
    w=None,
    attn_mask=None,
):
    if not (q.dim() == k.dim() == v.dim() == 3):
        if not (k.dim() == v.dim() == 4):
            raise ValueError(
                "k and v must both be 4-dimensional for multi-view"
                " cross-attention"
            )
        if not (h is not None and w is not None):
            raise ValueError(
                "h and w must be specified for multi-view cross-attention"
            )
        if q.size(1) != h * w:
            raise ValueError(f"q.size(1) must equal h*w={h*w}, got {q.size(1)}")

        m = k.size(1)

        b, _, c = q.size()

        b_new = b * num_splits * num_splits

        window_size_h = h // num_splits
        window_size_w = w // num_splits

        q = q.view(b, h, w, c)
        k = k.view(b, m, h, w, c)
        v = v.view(b, m, h, w, c)

        scale_factor = c**0.5

        if with_shift:
            if attn_mask is None:
                raise ValueError(
                    "attn_mask must be provided when with_shift is True"
                )
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(
                q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2)
            )
            k = torch.roll(
                k, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3)
            )
            v = torch.roll(
                v, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3)
            )

        q = split_feature(q, num_splits=num_splits, channel_last=True)
        k = split_feature(
            k.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )
        v = split_feature(
            v.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )

        k = (
            k.view(b_new, h // num_splits, w // num_splits, c, m)
            .permute(0, 3, 1, 2, 4)
            .reshape(b_new, c, -1)
        )
        v = (
            v.view(b_new, h // num_splits, w // num_splits, c, m)
            .permute(0, 1, 2, 4, 3)
            .reshape(b_new, -1, c)
        )

        scores = torch.matmul(q.view(b_new, -1, c), k) / scale_factor

        if with_shift:
            scores += attn_mask.repeat(b, 1, m)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)

        out = merge_splits(
            out.view(b_new, h // num_splits, w // num_splits, c),
            num_splits=num_splits,
            channel_last=True,
        )

        if with_shift:
            out = torch.roll(
                out, shifts=(shift_size_h, shift_size_w), dims=(1, 2)
            )

        out = out.view(b, -1, c)
    else:
        if not (q.dim() == k.dim() == v.dim() == 3):
            raise ValueError("q, k, v must all be 3-dimensional tensors")

        if not (h is not None and w is not None):
            raise ValueError("h and w must be specified")
        if q.size(1) != h * w:
            raise ValueError(f"q.size(1) must equal h*w={h*w}, got {q.size(1)}")

        b, _, c = q.size()

        b_new = b * num_splits * num_splits

        window_size_h = h // num_splits
        window_size_w = w // num_splits

        q = q.view(b, h, w, c)
        k = k.view(b, h, w, c)
        v = v.view(b, h, w, c)

        scale_factor = c**0.5

        if with_shift:
            if attn_mask is None:
                raise ValueError(
                    "attn_mask must be provided when with_shift is True"
                )
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(
                q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2)
            )
            k = torch.roll(
                k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2)
            )
            v = torch.roll(
                v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2)
            )

        q = split_feature(q, num_splits=num_splits, channel_last=True)
        k = split_feature(k, num_splits=num_splits, channel_last=True)
        v = split_feature(v, num_splits=num_splits, channel_last=True)

        scores = (
            torch.matmul(
                q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)
            )
            / scale_factor
        )

        if with_shift:
            scores += attn_mask.repeat(b, 1, 1)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.view(b_new, -1, c))

        out = merge_splits(
            out.view(b_new, h // num_splits, w // num_splits, c),
            num_splits=num_splits,
            channel_last=True,
        )

        if with_shift:
            out = torch.roll(
                out, shifts=(shift_size_h, shift_size_w), dims=(1, 2)
            )

        out = out.view(b, -1, c)

    return out


def multi_head_split_window_attention(
    q,
    k,
    v,
    num_splits=1,
    with_shift=False,
    h=None,
    w=None,
    attn_mask=None,
    num_head=1,
):
    if not (h is not None and w is not None):
        raise ValueError("h and w must be specified")
    if q.size(1) != h * w:
        raise ValueError(f"q.size(1) must equal h*w={h*w}, got {q.size(1)}")

    b, _, c = q.size()

    b_new = b * num_splits * num_splits

    window_size_h = h // num_splits
    window_size_w = w // num_splits

    q = q.view(b, h, w, c)
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    if c % num_head != 0:
        raise ValueError(
            f"channels {c} must be divisible by num_head {num_head}"
        )

    scale_factor = (c // num_head) ** 0.5

    if with_shift:
        if attn_mask is None:
            raise ValueError(
                "attn_mask must be provided when with_shift is True"
            )
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    q = split_feature(q, num_splits=num_splits)
    k = split_feature(k, num_splits=num_splits)
    v = split_feature(v, num_splits=num_splits)

    q = q.view(b_new, -1, num_head, c // num_head).permute(0, 2, 1, 3)
    k = k.view(b_new, -1, num_head, c // num_head).permute(0, 2, 3, 1)
    scores = torch.matmul(q, k) / scale_factor

    if with_shift:
        scores += attn_mask.unsqueeze(1).repeat(b, num_head, 1, 1)

    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(
        attn, v.view(b_new, -1, num_head, c // num_head).permute(0, 2, 1, 3)
    )

    out = merge_splits(
        out.permute(0, 2, 1, 3).reshape(
            b_new, h // num_splits, w // num_splits, c
        ),
        num_splits=num_splits,
    )

    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

    out = out.view(b, -1, c)

    return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=1,
        attention_type="swin",
        no_ffn=False,
        ffn_dim_expansion=4,
        with_shift=False,
        add_per_view_attn=False,
        **kwargs,
    ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.attention_type = attention_type
        self.no_ffn = no_ffn
        self.add_per_view_attn = add_per_view_attn

        self.with_shift = with_shift

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(
                    in_channels, in_channels * ffn_dim_expansion, bias=False
                ),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        source,
        target,
        height=None,
        width=None,
        shifted_window_attn_mask=None,
        attn_num_splits=None,
        **kwargs,
    ):
        if "attn_type" in kwargs:
            attn_type = kwargs["attn_type"]
        else:
            attn_type = self.attention_type

        query, key, value = source, target, target

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        if attn_type == "swin" and attn_num_splits > 1:
            if self.nhead > 1:
                message = multi_head_split_window_attention(
                    query,
                    key,
                    value,
                    num_splits=attn_num_splits,
                    with_shift=self.with_shift,
                    h=height,
                    w=width,
                    attn_mask=shifted_window_attn_mask,
                    num_head=self.nhead,
                )
            else:
                if self.add_per_view_attn:
                    if not (
                        query.dim() == 3 and key.dim() == 4 and value.dim() == 4
                    ):
                        raise ValueError(
                            "query must be 3D and key, value must be 4D for"
                            " per-view attention"
                        )
                    b, l, c = query.size()
                    query = query.unsqueeze(1).repeat(1, key.size(1), 1, 1)
                    query = query.view(-1, l, c)
                    key = key.view(-1, l, c)
                    value = value.view(-1, l, c)
                    message = single_head_split_window_attention(
                        query,
                        key,
                        value,
                        num_splits=attn_num_splits,
                        with_shift=self.with_shift,
                        h=height,
                        w=width,
                        attn_mask=shifted_window_attn_mask,
                    )
                    message = message.view(b, -1, l, c).sum(1)
                else:
                    message = single_head_split_window_attention(
                        query,
                        key,
                        value,
                        num_splits=attn_num_splits,
                        with_shift=self.with_shift,
                        h=height,
                        w=width,
                        attn_mask=shifted_window_attn_mask,
                    )
        else:
            message = single_head_full_attention(query, key, value)

        message = self.merge(message)
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(
        self,
        d_model=256,
        nhead=1,
        attention_type="swin",
        ffn_dim_expansion=4,
        with_shift=False,
        add_per_view_attn=False,
        no_cross_attn=False,
        **kwargs,
    ):
        super(TransformerBlock, self).__init__()

        self.no_cross_attn = no_cross_attn

        if no_cross_attn:
            self.self_attn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
                add_per_view_attn=add_per_view_attn,
            )
        else:
            self.self_attn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                no_ffn=True,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
            )

            self.cross_attn_ffn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
                add_per_view_attn=add_per_view_attn,
            )

    def forward(
        self,
        source,
        target,
        height=None,
        width=None,
        shifted_window_attn_mask=None,
        attn_num_splits=None,
        **kwargs,
    ):
        source = self.self_attn(
            source,
            source,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
            **kwargs,
        )

        if self.no_cross_attn:
            return source

        source = self.cross_attn_ffn(
            source,
            target,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
            **kwargs,
        )

        return source


def batch_features(features, nn_matrix=None):
    q = []
    kv = []

    num_views = len(features)
    if nn_matrix is not None:
        features_tensor = torch.stack(features, dim=1)

    for i in range(num_views):
        x = features.copy()
        q.append(x.pop(i))

        if nn_matrix is not None:
            if features_tensor.dim() == 5:
                c, h, w = features_tensor.shape[-3:]
                index = repeat(
                    nn_matrix[:, i, 1:], "b v -> b v c h w", c=c, h=h, w=w
                )
            elif features_tensor.dim() == 4:
                hw, c = features_tensor.shape[-2:]
                index = repeat(
                    nn_matrix[:, i, 1:], "b v -> b v hw c", hw=hw, c=c
                )

            kv_x = torch.gather(features_tensor, dim=1, index=index)
        else:
            kv_x = torch.stack(x, dim=1)
        kv.append(kv_x)

    q = torch.cat(q, dim=0)
    kv = torch.cat(kv, dim=0)

    return q, kv


class MultiViewFeatureTransformer(nn.Module):
    def __init__(
        self,
        num_layers=6,
        d_model=128,
        nhead=1,
        attention_type="swin",
        ffn_dim_expansion=4,
        add_per_view_attn=False,
        no_cross_attn=False,
        **kwargs,
    ):
        super(MultiViewFeatureTransformer, self).__init__()

        self.attention_type = attention_type

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    attention_type=attention_type,
                    ffn_dim_expansion=ffn_dim_expansion,
                    with_shift=(
                        True
                        if attention_type == "swin" and i % 2 == 1
                        else False
                    ),
                    add_per_view_attn=add_per_view_attn,
                    no_cross_attn=no_cross_attn,
                )
                for i in range(num_layers)
            ]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if num_layers > 6:
            for i in range(6, num_layers):
                self.layers[i].self_attn.norm1.weight.data.zero_()
                self.layers[i].self_attn.norm1.bias.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.weight.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.bias.data.zero_()

    def forward(
        self,
        multi_view_features,
        attn_num_splits=None,
        **kwargs,
    ):
        nn_matrix = kwargs.pop("nn_matrix", None)

        b, c, h, w = multi_view_features[0].shape
        if self.d_model != c:
            raise ValueError(
                f"d_model {self.d_model} must match feature channels {c}"
            )

        num_views = len(multi_view_features)

        if self.attention_type == "swin" and attn_num_splits > 1:
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=multi_view_features[0].device,
            )
        else:
            shifted_window_attn_mask = None

        concat0, concat1 = batch_features(
            multi_view_features, nn_matrix=nn_matrix
        )
        concat0 = concat0.reshape(num_views * b, c, -1).permute(0, 2, 1)
        c1_v = num_views - 1 if nn_matrix is None else nn_matrix.shape[-1] - 1
        concat1 = concat1.reshape(num_views * b, c1_v, c, -1).permute(
            0, 1, 3, 2
        )

        for i, layer in enumerate(self.layers):
            concat0 = layer(
                concat0,
                concat1,
                height=h,
                width=w,
                shifted_window_attn_mask=shifted_window_attn_mask,
                attn_num_splits=attn_num_splits,
            )

            if i < len(self.layers) - 1:
                features = list(concat0.chunk(chunks=num_views, dim=0))
                concat0, concat1 = batch_features(features, nn_matrix=nn_matrix)

        features = concat0.chunk(chunks=num_views, dim=0)
        features = [
            f.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            for f in features
        ]

        return features


def batch_features_camera_parameters(
    features,
    intrinsics,
    poses,
    nn_matrix=None,
    no_batch=False,
):
    if not (
        features[0].dim() == 4
        and intrinsics[0].dim() == 3
        and poses[0].dim() == 3
    ):
        raise ValueError("features must be 4D, intrinsics and poses must be 3D")
    if not (intrinsics[0].size(-1) == intrinsics[0].size(-2) == 3):
        raise ValueError("intrinsics must be 3x3 matrices")
    if not (poses[0].size(-1) == poses[0].size(-2) == 4):
        raise ValueError("poses must be 4x4 matrices")

    q = []
    q_intrinsics = []
    q_poses = []
    kv = []
    kv_intrinsics = []
    kv_poses = []

    num_views = len(features)
    if nn_matrix is not None:
        features_tensor = torch.stack(features, dim=1)
        intrinsics_tensor = torch.stack(intrinsics, dim=1)
        poses_tensor = torch.stack(poses, dim=1)

        num_selected_views = nn_matrix.size(-1) - 1
    else:
        num_selected_views = num_views - 1

    for i in range(num_views):
        x = features.copy()
        q.append(x.pop(i))

        y = intrinsics.copy()
        q_intrinsics.append(y.pop(i))
        z = poses.copy()
        q_poses.append(z.pop(i))

        if nn_matrix is not None:
            if features_tensor.dim() == 5:
                c, h, w = features_tensor.shape[-3:]
                index = repeat(
                    nn_matrix[:, i, 1:], "b v -> b v c h w", c=c, h=h, w=w
                )
            elif features_tensor.dim() == 4:
                hw, c = features_tensor.shape[-2:]
                index = repeat(
                    nn_matrix[:, i, 1:], "b v -> b v hw c", hw=hw, c=c
                )

            kv_x = torch.gather(features_tensor, dim=1, index=index)

            index = repeat(nn_matrix[:, i, 1:], "b v -> b v 3 3")
            kv_y_intrinsics = torch.gather(
                intrinsics_tensor, dim=1, index=index
            )

            index = repeat(nn_matrix[:, i, 1:], "b v -> b v 4 4")
            kv_z_poses = torch.gather(poses_tensor, dim=1, index=index)

        else:
            kv_x = torch.stack(x, dim=1)
            kv_y_intrinsics = torch.stack(y, dim=1)
            kv_z_poses = torch.stack(z, dim=1)

        kv.append(kv_x)
        kv_intrinsics.append(kv_y_intrinsics)
        kv_poses.append(kv_z_poses)

    if no_batch:
        return q, q_intrinsics, q_poses, kv, kv_intrinsics, kv_poses

    c, h, w = q[0].shape[1:]

    q = torch.stack(q, dim=1).view(-1, c, h, w)
    q_intrinsics = torch.stack(q_intrinsics, dim=1).view(-1, 3, 3)
    q_poses = torch.stack(q_poses, dim=1).view(-1, 4, 4)
    kv = torch.stack(kv, dim=1).view(-1, num_selected_views, c, h, w)
    kv_intrinsics = torch.stack(kv_intrinsics, dim=1).view(
        -1, num_selected_views, 3, 3
    )
    kv_poses = torch.stack(kv_poses, dim=1).view(-1, num_selected_views, 4, 4)

    return q, q_intrinsics, q_poses, kv, kv_intrinsics, kv_poses


class MultiViewUniMatch(nn.Module):
    def __init__(
        self,
        num_scales=1,
        feature_channels=128,
        upsample_factor=8,
        lowest_feature_resolution=8,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        num_depth_candidates=128,
        vit_type="vits",
        unet_channels=128,
        unet_channel_mult=[1, 1, 1],
        unet_num_res_blocks=1,
        unet_attn_resolutions=[4],
        grid_sample_disable_cudnn=False,
        **kwargs,
    ):
        super(MultiViewUniMatch, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.lowest_feature_resolution = lowest_feature_resolution
        self.upsample_factor = upsample_factor

        self.vit_type = vit_type

        self.num_depth_candidates = num_depth_candidates

        vit_feature_channel_dict = {"vits": 384, "vitb": 768, "vitl": 1024}

        vit_feature_channel = vit_feature_channel_dict[vit_type]

        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=num_scales,
            downsample_factor=upsample_factor,
            lowest_scale=lowest_feature_resolution,
            return_all_scales=True,
        )

        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )

        if self.num_scales > 1:
            self.mv_pyramid = ViTFeaturePyramid(
                in_channels=128,
                scale_factors=[2**i for i in range(self.num_scales)],
            )

        encoder = vit_type
        self.pretrained = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_{:}14".format(encoder)
        )

        del self.pretrained.mask_token

        if self.num_scales > 1:
            self.mono_pyramid = ViTFeaturePyramid(
                in_channels=vit_feature_channel,
                scale_factors=[2**i for i in range(self.num_scales)],
            )

        self.regressor = nn.ModuleList()
        self.regressor_residual = nn.ModuleList()
        self.depth_head = nn.ModuleList()

        for i in range(self.num_scales):
            curr_depth_candidates = num_depth_candidates // (4**i)
            cnn_feature_channels = 128 - (32 * i)
            mv_transformer_feature_channels = 128 // (2**i)

            mono_feature_channels = vit_feature_channel // (2**i)

            in_channels = (
                curr_depth_candidates
                + cnn_feature_channels
                + mv_transformer_feature_channels
                + mono_feature_channels
            )

            channels = unet_channels // (2**i)

            if i > 0:
                unet_channel_mult = unet_channel_mult + [1]
                unet_attn_resolutions = [x * 2 for x in unet_attn_resolutions]

            modules = [
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ]

            modules.append(
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=unet_num_res_blocks,
                    attention_resolutions=unet_attn_resolutions,
                    channel_mult=unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=False,
                    num_frames=2,
                    use_cross_view_self_attn=True,
                )
            )

            modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

            self.regressor.append(nn.Sequential(*modules))

            self.regressor_residual.append(nn.Conv2d(in_channels, channels, 1))

            self.depth_head.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels * 2,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        channels * 2,
                        curr_depth_candidates,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                    ),
                )
            )

        in_channels = (
            1
            + cnn_feature_channels
            + mv_transformer_feature_channels
            + mono_feature_channels
        )

        model_configs = {
            "vits": {
                "in_channels": 384,
                "features": 32,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "in_channels": 768,
                "features": 48,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "in_channels": 1024,
                "features": 64,
                "out_channels": [128, 256, 512, 1024],
            },
        }

        self.upsampler = DPTHead(
            **model_configs[vit_type],
            downsample_factor=upsample_factor,
            num_scales=num_scales,
        )

        self.grid_sample_disable_cudnn = grid_sample_disable_cudnn

    def normalize_images(self, images):
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = (
            torch.tensor([0.485, 0.456, 0.406])
            .reshape(*shape)
            .to(images.device)
        )
        std = (
            torch.tensor([0.229, 0.224, 0.225])
            .reshape(*shape)
            .to(images.device)
        )

        return (images - mean) / std

    def extract_feature(self, images):
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        features = self.backbone(concat)
        features = features[::-1]

        return features

    def forward(
        self,
        images,
        attn_splits_list=None,
        intrinsics=None,
        min_depth=1.0 / 0.5,
        max_depth=1.0 / 100,
        num_depth_candidates=128,
        poses=None,
        nn_matrix=None,
        **kwargs,
    ):
        results_dict = {}
        depth_preds = []
        match_probs = []

        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape

        set_num_views(self.regressor, num_views=v)

        intrinsics = intrinsics.clone()
        intrinsics[:, :, 0] *= ori_w
        intrinsics[:, :, 1] *= ori_h

        max_depth = max_depth.view(-1)
        min_depth = min_depth.view(-1)

        features_list_cnn = self.extract_feature(images)
        features_list_cnn_all_scales = features_list_cnn
        features_list_cnn = features_list_cnn[: self.num_scales]
        results_dict.update(
            {"features_cnn_all_scales": features_list_cnn_all_scales}
        )
        results_dict.update({"features_cnn": features_list_cnn})

        attn_splits = attn_splits_list[0]

        features_cnn_pos = mv_feature_add_position(
            features_list_cnn[0], attn_splits, self.feature_channels
        )

        features_list = list(
            torch.unbind(
                rearrange(
                    features_cnn_pos, "(b v) c h w -> b v c h w", b=b, v=v
                ),
                dim=1,
            )
        )
        features_list_mv = self.transformer(
            features_list,
            attn_num_splits=attn_splits,
            nn_matrix=nn_matrix,
        )

        features_mv = rearrange(
            torch.stack(features_list_mv, dim=1), "b v c h w -> (b v) c h w"
        )

        if self.num_scales > 1:
            features_list_mv = self.mv_pyramid(features_mv)
        else:
            features_list_mv = [features_mv]

        results_dict.update({"features_mv": features_list_mv})

        ori_h, ori_w = images.shape[-2:]
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        concat = F.interpolate(
            concat, (resize_h, resize_w), mode="bilinear", align_corners=True
        )

        intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }

        mono_intermediate_features = list(
            self.pretrained.get_intermediate_layers(
                concat,
                intermediate_layer_idx[self.vit_type],
                return_class_token=False,
            )
        )

        for i in range(len(mono_intermediate_features)):
            curr_features = (
                mono_intermediate_features[i]
                .reshape(concat.shape[0], resize_h // 14, resize_w // 14, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            curr_features = F.interpolate(
                curr_features,
                (ori_h // 8, ori_w // 8),
                mode="bilinear",
                align_corners=True,
            )
            mono_intermediate_features[i] = curr_features

        results_dict.update(
            {"features_mono_intermediate": mono_intermediate_features}
        )

        mono_features = mono_intermediate_features[-1]

        if self.lowest_feature_resolution == 4:
            mono_features = F.interpolate(
                mono_features,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )

        if self.num_scales > 1:
            features_list_mono = self.mono_pyramid(mono_features)
        else:
            features_list_mono = [mono_features]

        results_dict.update({"features_mono": features_list_mono})

        depth = None

        for scale_idx in range(self.num_scales):
            downsample_factor = self.upsample_factor * (
                2 ** (self.num_scales - 1 - scale_idx)
            )

            intrinsics_curr = intrinsics.clone()
            intrinsics_curr[:, :, :2] = (
                intrinsics_curr[:, :, :2] / downsample_factor
            )

            features_mv = features_list_mv[scale_idx]

            features_mv_curr = list(
                torch.unbind(
                    rearrange(
                        features_mv, "(b v) c h w -> b v c h w", b=b, v=v
                    ),
                    dim=1,
                )
            )

            intrinsics_curr = list(torch.unbind(intrinsics_curr, dim=1))
            poses_curr = list(torch.unbind(poses, dim=1))

            (
                ref_features,
                ref_intrinsics,
                ref_poses,
                tgt_features,
                tgt_intrinsics,
                tgt_poses,
            ) = batch_features_camera_parameters(
                features_mv_curr,
                intrinsics_curr,
                poses_curr,
                nn_matrix=nn_matrix,
            )

            b_new, _, c, h, w = tgt_features.size()

            pose_curr = torch.matmul(
                tgt_poses.inverse(), ref_poses.unsqueeze(1)
            )

            if scale_idx > 0:
                if depth is None:
                    raise ValueError("depth must not be None for scale_idx > 0")
                depth = F.interpolate(
                    depth, scale_factor=2, mode="bilinear", align_corners=True
                ).detach()

            num_depth_candidates = self.num_depth_candidates // (4**scale_idx)

            if scale_idx == 0:
                depth_interval = (max_depth - min_depth) / (
                    self.num_depth_candidates - 1
                )

                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )

                depth_candidates = min_depth.view(
                    -1, 1, 1, 1
                ) + linear_space * (max_depth - min_depth).view(-1, 1, 1, 1)
            else:
                depth_interval = (
                    (max_depth - min_depth)
                    / (self.num_depth_candidates - 1)
                    / (2**scale_idx)
                )
                depth_interval = depth_interval.view(-1, 1, 1, 1)

                depth_range_min = (
                    depth - depth_interval * (num_depth_candidates // 2)
                ).clamp(min=min_depth.view(-1, 1, 1, 1))
                depth_range_max = (
                    depth + depth_interval * (num_depth_candidates // 2 - 1)
                ).clamp(max=max_depth.view(-1, 1, 1, 1))

                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )
                depth_candidates = depth_range_min + linear_space * (
                    depth_range_max - depth_range_min
                )

            if scale_idx == 0:
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, h, w)
                    .view(-1, num_depth_candidates, h, w)
                )
            else:
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, 1, 1)
                    .view(-1, num_depth_candidates, h, w)
                )

            intrinsics_input = torch.stack(intrinsics_curr, dim=1).view(
                -1, 3, 3
            )
            intrinsics_input = intrinsics_input.unsqueeze(1).repeat(
                1, tgt_features.size(1), 1, 1
            )

            warped_tgt_features = warp_with_pose_depth_candidates(
                rearrange(tgt_features, "b v ... -> (b v) ..."),
                rearrange(intrinsics_input, "b v ... -> (b v) ..."),
                rearrange(pose_curr, "b v ... -> (b v) ..."),
                1.0 / depth_candidates_curr,
                grid_sample_disable_cudnn=self.grid_sample_disable_cudnn,
            )

            warped_tgt_features = rearrange(
                warped_tgt_features,
                "(b v) ... -> b v ...",
                b=b_new,
                v=tgt_features.size(1),
            )
            cost_volume = (
                (
                    ref_features.unsqueeze(-3).unsqueeze(1)
                    * warped_tgt_features
                ).sum(2)
                / (c**0.5)
            ).mean(1)

            features_cnn = features_list_cnn[scale_idx]

            features_mono = features_list_mono[scale_idx]

            concat = torch.cat(
                (cost_volume, features_cnn, features_mv, features_mono), dim=1
            )

            out = self.regressor[scale_idx](concat) + self.regressor_residual[
                scale_idx
            ](concat)

            match_prob = F.softmax(self.depth_head[scale_idx](out), dim=1)
            match_probs.append(match_prob)

            if scale_idx == 0:
                depth_candidates = depth_candidates.repeat(1, 1, h, w)
            depth = (match_prob * depth_candidates).sum(dim=1, keepdim=True)

            if self.training and scale_idx < self.num_scales - 1:
                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=downsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth_preds.append(depth_bilinear)

            if scale_idx == self.num_scales - 1:
                residual_depth = self.upsampler(
                    mono_intermediate_features,
                    cnn_features=features_list_cnn_all_scales[::-1],
                    mv_features=(
                        features_mv
                        if self.num_scales == 1
                        else features_list_mv[::-1]
                    ),
                    depth=depth,
                )

                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=self.upsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth = (depth_bilinear + residual_depth).clamp(
                    min=min_depth.view(-1, 1, 1, 1),
                    max=max_depth.view(-1, 1, 1, 1),
                )

                depth_preds.append(depth)

        for i in range(len(depth_preds)):
            depth_pred = 1.0 / depth_preds[i].squeeze(1)
            depth_preds[i] = rearrange(
                depth_pred, "(b v) ... -> b v ...", b=b, v=v
            )

        results_dict.update({"depth_preds": depth_preds})
        results_dict.update({"match_probs": match_probs})

        return results_dict


def set_num_views(module, num_views):
    if isinstance(module, AttentionBlock):
        module.attention.n_frames = num_views
    elif (
        isinstance(module, nn.ModuleList)
        or isinstance(module, nn.Sequential)
        or isinstance(module, nn.Module)
    ):
        for submodule in module.children():
            set_num_views(submodule, num_views)
