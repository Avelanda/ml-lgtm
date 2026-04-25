"""
NoPoSplat backbone.

For third-party code see ACKNOWLEDGMENTS file.

By using the NoPoSplat backbone, you agree to comply with the licenses of
NoPoSplat and all associated dependencies, including but not limited to:
- noposplat: https://github.com/cvg/NoPoSplat/blob/main/LICENSE
- mast3r: https://github.com/naver/mast3r/blob/main/LICENSE
- croco: https://github.com/naver/croco/blob/master/LICENSE
"""

import collections.abc
import inspect
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from typing import Generic, Literal, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from lgtm.dataset.data_types import BatchedViews
from lgtm.utils.geometry_utils import get_intrinsic_embedding

torch.backends.cuda.matmul.allow_tf32 = True


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(
    x,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        rope=None,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .transpose(1, 3)
        )
        q, k, v = [qkv[:, :, i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # Make torchscript happy (cannot use tensor as tuple).

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        # q, k, v shape: (B, num_heads, N, head_dim)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        # Output shape: (B, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here.
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        rope=None,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(self, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = (
            self.projq(query)
            .reshape(B, Nq, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.projk(key)
            .reshape(B, Nk, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.projv(value)
            .reshape(B, Nv, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        # q, k, v shape: (B, num_heads, N, head_dim)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        # Output shape: (B, num_heads, Nq, head_dim)
        x = x.transpose(1, 2).reshape(B, Nq, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.cross_attn = CrossAttention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y)
        x = x + self.drop_path(
            self.cross_attn(self.norm2(x), y_, y_, xpos, ypos)
        )
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y


class PositionGetter(object):
    """return positions of patches"""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if (h, w) not in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h, w] = torch.cartesian_prod(y, x)  # (h, w, 2)
        pos = (
            self.cache_positions[h, w]
            .view(1, h * w, 2)
            .expand(b, -1, 2)
            .clone()
        )
        return pos


class PatchEmbed(nn.Module):
    """just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.position_getter = PositionGetter()

    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model"
            f" ({self.img_size[0]}).",
        )
        torch._assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model"
            f" ({self.img_size[1]}).",
        )
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos

    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


class RandomMask(nn.Module):
    """
    random masking
    """

    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)

    def __call__(self, x):
        noise = torch.rand(x.size(0), self.num_patches, device=x.device)
        argsort = torch.argsort(noise, dim=1)
        return argsort < self.num_mask


def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token > 0:
        pos_embed = np.concatenate(
            [np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")

    # Use half of dimensions to encode grid_h.
    emb_h = _get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


_CUROPE_CPP_SOURCE = """
/*
  Copyright (C) 2022-present Naver Corporation. All rights reserved.
  Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
*/

#include <torch/extension.h>

// forward declaration
void rope_2d_cuda(torch::Tensor tokens, const torch::Tensor pos,
                  const float base, const float fwd);

void rope_2d_cpu(torch::Tensor tokens, const torch::Tensor positions,
                 const float base, const float fwd) {
  const int B = tokens.size(0);
  const int N = tokens.size(1);
  const int H = tokens.size(2);
  const int D = tokens.size(3) / 4;

  auto tok = tokens.accessor<float, 4>();
  auto pos = positions.accessor<int64_t, 3>();

  for (int b = 0; b < B; b++) {
    for (int x = 0; x < 2; x++) { // y and then x (2d)
      for (int n = 0; n < N; n++) {

        // grab the token position
        const int p = pos[b][n][x];

        for (int h = 0; h < H; h++) {
          for (int d = 0; d < D; d++) {
            // grab the two values
            float u = tok[b][n][h][d + 0 + x * 2 * D];
            float v = tok[b][n][h][d + D + x * 2 * D];

            // grab the cos,sin
            const float inv_freq = fwd * p / powf(base, d / float(D));
            float c = cosf(inv_freq);
            float s = sinf(inv_freq);

            // write the result
            tok[b][n][h][d + 0 + x * 2 * D] = u * c - v * s;
            tok[b][n][h][d + D + x * 2 * D] = v * c + u * s;
          }
        }
      }
    }
  }
}

void rope_2d(torch::Tensor tokens,          // B,N,H,D
             const torch::Tensor positions, // B,N,2
             const float base, const float fwd) {
  TORCH_CHECK(tokens.dim() == 4, "tokens must have 4 dimensions");
  TORCH_CHECK(positions.dim() == 3, "positions must have 3 dimensions");
  TORCH_CHECK(tokens.size(0) == positions.size(0),
              "batch size differs between tokens & positions");
  TORCH_CHECK(tokens.size(1) == positions.size(1),
              "seq_length differs between tokens & positions");
  TORCH_CHECK(positions.size(2) == 2, "positions.shape[2] must be equal to 2");
  TORCH_CHECK(tokens.is_cuda() == positions.is_cuda(),
              "tokens and positions are not on the same device");

  if (tokens.is_cuda())
    rope_2d_cuda(tokens, positions, base, fwd);
  else
    rope_2d_cpu(tokens, positions, base, fwd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_2d", &rope_2d, "RoPE 2d forward/backward");
}
"""

_CUROPE_CUDA_SOURCE = """
/*
  Copyright (C) 2022-present Naver Corporation. All rights reserved.
  Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(tensor)                                                     \\
  {                                                                            \\
    TORCH_CHECK((tensor).is_cuda(), #tensor " is not in cuda memory");         \\
    TORCH_CHECK((tensor).is_contiguous(), #tensor " is not contiguous");       \\
  }
void CHECK_KERNEL() {
  auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

template <typename scalar_t>
__global__ void rope_2d_cuda_kernel(
    // scalar_t* __restrict__ tokens,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> tokens,
    const int64_t *__restrict__ pos, const float base, const float fwd)
// const int N, const int H, const int D )
{
  // tokens shape = (B, N, H, D)
  const int N = tokens.size(1);
  const int H = tokens.size(2);
  const int D = tokens.size(3);

  // each block update a single token, for all heads
  // each thread takes care of a single output
  extern __shared__ float shared[];
  float *shared_inv_freq = shared + D;

  const int b = blockIdx.x / N;
  const int n = blockIdx.x % N;

  const int Q = D / 4;
  // one token = [0..Q : Q..2Q : 2Q..3Q : 3Q..D]
  //              u_Y     v_Y     u_X      v_X

  // shared memory: first, compute inv_freq
  if (threadIdx.x < Q)
    shared_inv_freq[threadIdx.x] = fwd / powf(base, threadIdx.x / float(Q));
  __syncthreads();

  // start of X or Y part
  const int X = threadIdx.x < D / 2 ? 0 : 1;
  const int m = (X * D / 2) + (threadIdx.x % Q); // index of u_Y or u_X

  // grab the cos,sin appropriate for me
  const float freq = pos[blockIdx.x * 2 + X] * shared_inv_freq[threadIdx.x % Q];
  const float cos = cosf(freq);
  const float sin = sinf(freq);
  /*
  float* shared_cos_sin = shared + D + D/4;
  if ((threadIdx.x % (D/2)) < Q)
      shared_cos_sin[m+0] = cosf(freq);
  else
      shared_cos_sin[m+Q] = sinf(freq);
  __syncthreads();
  const float cos = shared_cos_sin[m+0];
  const float sin = shared_cos_sin[m+Q];
  */

  for (int h = 0; h < H; h++) {
    // then, load all the token for this head in shared memory
    shared[threadIdx.x] = tokens[b][n][h][threadIdx.x];
    __syncthreads();

    const float u = shared[m];
    const float v = shared[m + Q];

    // write output
    if ((threadIdx.x % (D / 2)) < Q)
      tokens[b][n][h][threadIdx.x] = u * cos - v * sin;
    else
      tokens[b][n][h][threadIdx.x] = v * cos + u * sin;
  }
}

void rope_2d_cuda(torch::Tensor tokens, const torch::Tensor pos,
                  const float base, const float fwd) {
  const int B = tokens.size(0); // batch size
  const int N = tokens.size(1); // sequence length
  const int H = tokens.size(2); // number of heads
  const int D = tokens.size(3); // dimension per head

  TORCH_CHECK(tokens.stride(3) == 1 && tokens.stride(2) == D,
              "tokens are not contiguous");
  TORCH_CHECK(pos.is_contiguous(), "positions are not contiguous");
  TORCH_CHECK(pos.size(0) == B && pos.size(1) == N && pos.size(2) == 2,
              "bad pos.shape");
  TORCH_CHECK(D % 4 == 0, "token dim must be multiple of 4");

  // one block for each layer, one thread per local-max
  const int THREADS_PER_BLOCK = D;
  const int N_BLOCKS = B * N; // each block takes care of H*D values
  const int SHARED_MEM = sizeof(float) * (D + D / 4);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      tokens.scalar_type(), "rope_2d_cuda", ([&] {
        rope_2d_cuda_kernel<scalar_t>
            <<<N_BLOCKS, THREADS_PER_BLOCK, SHARED_MEM>>>(
                // tokens.data_ptr<scalar_t>(),
                tokens
                    .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                pos.data_ptr<int64_t>(), base, fwd); //, N, H, D );
      }));
}
"""

_curope_kernels = None


def _get_curope_kernels():
    """
    Lazily compile and cache the cuRoPE CUDA kernels on first call.

    PyTorch's load_inline uses FileBaton (file locking) internally, so
    concurrent calls from multiple DDP processes are safe: one process
    compiles while the others wait, then all load the cached result.
    """
    global _curope_kernels
    if _curope_kernels is not None:
        return _curope_kernels

    import os
    import time

    from torch.utils.cpp_extension import load_inline as _load_inline

    rank = int(os.environ.get("LOCAL_RANK", 0))
    t0 = time.monotonic()
    _curope_kernels = _load_inline(
        name="curope",
        cpp_sources=[_CUROPE_CPP_SOURCE],
        cuda_sources=[_CUROPE_CUDA_SOURCE],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    elapsed = time.monotonic() - t0
    if rank == 0 and elapsed > 3.0:
        print(f"cuRoPE: compiled CUDA kernels in {elapsed:.1f}s.", flush=True)
    return _curope_kernels


def _is_curope_available():
    """
    Check if cuRoPE CUDA JIT compilation is possible without actually
    compiling. Returns False on systems without CUDA or a C++ compiler.
    """
    try:
        from torch.utils.cpp_extension import (
            load_inline as _load_inline,  # noqa: F401
        )

        return torch.cuda.is_available()
    except (ImportError, RuntimeError, OSError):
        return False


if _is_curope_available():

    class _cuRoPE2D_func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tokens, positions, base, F0=1):
            ctx.save_for_backward(positions)
            ctx.saved_base = base
            ctx.saved_F0 = F0
            # tokens = tokens.clone()  # Uncomment this if inplace does not work.
            _get_curope_kernels().rope_2d(tokens, positions, base, F0)
            ctx.mark_dirty(tokens)
            return tokens

        @staticmethod
        def backward(ctx, grad_res):
            positions, base, F0 = (
                ctx.saved_tensors[0],
                ctx.saved_base,
                ctx.saved_F0,
            )
            _get_curope_kernels().rope_2d(grad_res, positions, base, -F0)
            ctx.mark_dirty(grad_res)
            return grad_res, None, None, None

    class cuRoPE2D(torch.nn.Module):
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq
            self.F0 = F0

        def forward(self, tokens, positions):
            _cuRoPE2D_func.apply(
                tokens.transpose(1, 2), positions, self.base, self.F0
            )
            return tokens

    RoPE2D = cuRoPE2D
else:
    print(
        "Warning, cannot find cuda-compiled version of RoPE2D, using a slow"
        " pytorch version instead"
    )

    class RoPE2D(torch.nn.Module):
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D, seq_len, device, dtype) not in self.cache:
                inv_freq = 1.0 / (
                    self.base ** (torch.arange(0, D, 2).float().to(device) / D)
                )
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos()  # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D, seq_len, device, dtype] = (cos, sin)
            return self.cache[D, seq_len, device, dtype]

        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope1d(self, tokens, pos1d, cos, sin):
            if pos1d.ndim != 2:
                raise ValueError(
                    f"pos1d must be 2-dimensional, got {pos1d.ndim}"
                )
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)

        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            if tokens.size(3) % 2 != 0:
                raise ValueError(
                    "number of dimensions should be a multiple of two"
                )
            D = tokens.size(3) // 2
            if not (
                positions.ndim == 3 and positions.shape[-1] == 2
            ):  # Batch, Seq, 2
                raise ValueError(
                    "positions must be 3-dimensional with last dim == 2"
                )
            cos, sin = self.get_cos_sin(
                D, int(positions.max()) + 1, tokens.device, tokens.dtype
            )
            # Split features into two along the feature dimension, and apply rope1d on each half.
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
            x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens


def get_patch_embed(
    patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
):
    _PATCH_EMBED_MAP = {
        "PatchEmbedDust3R": PatchEmbedDust3R,
        "ManyAR_PatchEmbed": ManyAR_PatchEmbed,
    }
    if patch_embed_cls not in _PATCH_EMBED_MAP:
        raise ValueError(
            f"Unknown patch_embed_cls '{patch_embed_cls}', "
            f"expected one of {list(_PATCH_EMBED_MAP)}"
        )
    patch_embed = _PATCH_EMBED_MAP[patch_embed_cls](
        img_size, patch_size, in_chans, enc_embed_dim
    )
    return patch_embed


class PatchEmbedDust3R(PatchEmbed):
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        if H % self.patch_size[0] != 0:
            raise ValueError(
                f"Input image height ({H}) is not a multiple of patch size"
                f" ({self.patch_size[0]})."
            )
        if W % self.patch_size[1] != 0:
            raise ValueError(
                f"Input image width ({W}) is not a multiple of patch size"
                f" ({self.patch_size[1]})."
            )
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos


class ManyAR_PatchEmbed(PatchEmbed):
    """
    Handle images with non-square aspect ratio.
    All images in the same batch have the same aspect ratio.
    true_shape = [(height, width) ...] indicates the actual shape of each image.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        self.embed_dim = embed_dim
        super().__init__(
            img_size, patch_size, in_chans, embed_dim, norm_layer, flatten
        )

    def forward(self, img, true_shape):
        B, C, H, W = img.shape
        if W < H:
            raise ValueError(
                f"img should be in landscape mode, but got {W=} {H=}"
            )
        if H % self.patch_size[0] != 0:
            raise ValueError(
                f"Input image height ({H}) is not a multiple of patch size"
                f" ({self.patch_size[0]})."
            )
        if W % self.patch_size[1] != 0:
            raise ValueError(
                f"Input image width ({W}) is not a multiple of patch size"
                f" ({self.patch_size[1]})."
            )
        if true_shape.shape != (B, 2):
            raise ValueError(
                f"true_shape has the wrong shape={true_shape.shape}"
            )

        # Size expressed in tokens.
        W //= self.patch_size[0]
        H //= self.patch_size[1]
        n_tokens = H * W

        height, width = true_shape.T
        is_landscape = width >= height
        is_portrait = ~is_landscape

        # Allocate result.
        x = img.new_zeros((B, n_tokens, self.embed_dim))
        pos = img.new_zeros((B, n_tokens, 2), dtype=torch.int64)

        # Linear projection, transposed if necessary.
        x[is_landscape] = (
            self.proj(img[is_landscape])
            .permute(0, 2, 3, 1)
            .flatten(1, 2)
            .float()
        )
        x[is_portrait] = (
            self.proj(img[is_portrait].swapaxes(-1, -2))
            .permute(0, 2, 3, 1)
            .flatten(1, 2)
            .float()
        )

        pos[is_landscape] = self.position_getter(1, H, W, pos.device)
        pos[is_portrait] = self.position_getter(1, W, H, pos.device)

        x = self.norm(x)
        return x, pos


def fill_default_args(kwargs, func):
    signature = inspect.signature(func)

    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            continue
        kwargs.setdefault(k, v.default)

    return kwargs


def is_symmetrized(gt1, gt2):
    x = gt1["instance"]
    y = gt2["instance"]
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def interleave(tensor1, tensor2):
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(view1, view2):
    view1, view2 = (
        _interleave_imgs(view1, view2),
        _interleave_imgs(view2, view1),
    )
    return view1, view2


def transpose_to_landscape(head, activate=True):
    """
    Predict in the correct aspect-ratio,
    then transpose the result in landscape
    and stack everything back together.
    """

    def wrapper_no(decout, true_shape, ray_embedding=None):
        B = len(true_shape)
        if not true_shape[0:1].allclose(true_shape):
            raise ValueError("true_shape must be all identical")
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W), ray_embedding=ray_embedding)
        return res

    def wrapper_yes(decout, true_shape, ray_embedding=None):
        B = len(true_shape)
        # By definition, the batch is in landscape mode so W >= H.
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = width >= height
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W), ray_embedding=ray_embedding)
        if is_portrait.all():
            return transposed(head(decout, (W, H), ray_embedding=ray_embedding))

        # Batch is a mix of both portraint & landscape.
        def selout(ar):
            return [d[ar] for d in decout]

        l_result = head(
            selout(is_landscape), (H, W), ray_embedding=ray_embedding
        )
        p_result = transposed(
            head(selout(is_portrait), (W, H), ray_embedding=ray_embedding)
        )

        # Allocate full result.
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no


def transposed(dic):
    return {k: v.swapaxes(1, 2) for k, v in dic.items()}


class CroCoNet(nn.Module):
    def __init__(
        self,
        img_size=224,  # input image size
        patch_size=16,  # patch_size
        mask_ratio=0.9,  # ratios of masked tokens
        enc_embed_dim=768,  # encoder feature dimension
        enc_depth=12,  # encoder depth
        enc_num_heads=12,  # encoder number of heads in the transformer block
        dec_embed_dim=512,  # decoder feature dimension
        dec_depth=8,  # decoder depth
        dec_num_heads=16,  # decoder number of heads in the transformer block
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_im2_in_dec=True,  # whether to apply normalization of the 'memory' = (second image) in the decoder
        pos_embed="cosine",  # positional embedding (either cosine or RoPE100)
    ):

        super(CroCoNet, self).__init__()

        # Patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(img_size, patch_size, enc_embed_dim)

        # Mask generations.
        self._set_mask_generator(self.patch_embed.num_patches, mask_ratio)

        self.pos_embed = pos_embed
        if pos_embed == "cosine":
            # Positional embedding of the encoder.
            enc_pos_embed = get_2d_sincos_pos_embed(
                enc_embed_dim,
                int(self.patch_embed.num_patches**0.5),
                n_cls_token=0,
            )
            self.register_buffer(
                "enc_pos_embed", torch.from_numpy(enc_pos_embed).float()
            )
            # Positional embedding of the decoder.
            dec_pos_embed = get_2d_sincos_pos_embed(
                dec_embed_dim,
                int(self.patch_embed.num_patches**0.5),
                n_cls_token=0,
            )
            self.register_buffer(
                "dec_pos_embed", torch.from_numpy(dec_pos_embed).float()
            )
            # Pos embedding in each block.
            self.rope = None  # nothing for cosine
        elif pos_embed.startswith("RoPE"):  # eg RoPE100
            self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the"
                    " README instructions"
                )
            freq = float(pos_embed[len("RoPE") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + pos_embed)

        # Transformer for the encoder.
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList(
            [
                Block(
                    enc_embed_dim,
                    enc_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    rope=self.rope,
                )
                for i in range(enc_depth)
            ]
        )
        self.enc_norm = norm_layer(enc_embed_dim)

        # Masked tokens.
        self._set_mask_token(dec_embed_dim)

        # Decoder.
        self._set_decoder(
            enc_embed_dim,
            dec_embed_dim,
            dec_num_heads,
            dec_depth,
            mlp_ratio,
            norm_layer,
            norm_im2_in_dec,
        )

        # Prediction head.
        self._set_prediction_head(dec_embed_dim, patch_size)

        # Initializer weights.
        self.initialize_weights()

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    def _set_mask_generator(self, num_patches, mask_ratio):
        self.mask_generator = RandomMask(num_patches, mask_ratio)

    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # Transfer from encoder to decoder.
        self.decoder_embed = nn.Linear(
            enc_embed_dim + 0, dec_embed_dim, bias=True
        )
        # Transformer for the decoder.
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        # Final norm layer.
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_prediction_head(self, dec_embed_dim, patch_size):
        self.prediction_head = nn.Linear(
            dec_embed_dim, patch_size**2 * 3, bias=True
        )

    def initialize_weights(self):
        # Patch embed.
        self.patch_embed._init_weights()
        # Mask tokens.
        if self.mask_token is not None:
            torch.nn.init.normal_(self.mask_token, std=0.02)
        # Linears and layer norms.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _encode_image(self, image, do_mask=False, return_all_blocks=False):
        """
        image has B x 3 x img_size x img_size
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # Embed the image into patches  (x has size B x Npatches x C)
        # And get position if each return patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(image)
        # Add positional embedding without cls token.
        if self.enc_pos_embed is not None:
            x = x + self.enc_pos_embed[None, ...]
        # Apply masking.
        B, N, C = x.size()
        if do_mask:
            masks = self.mask_generator(x)
            x = x[~masks].view(B, -1, C)
            posvis = pos[~masks].view(B, -1, 2)
        else:
            B, N, C = x.size()
            masks = torch.zeros((B, N), dtype=bool)
            posvis = pos
        # Now apply the transformer encoder and normalization.
        if return_all_blocks:
            out = []
            for blk in self.enc_blocks:
                x = blk(x, posvis)
                out.append(x)
            out[-1] = self.enc_norm(out[-1])
            return out, pos, masks
        else:
            for blk in self.enc_blocks:
                x = blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos, masks

    def _decoder(
        self, feat1, pos1, masks1, feat2, pos2, return_all_blocks=False
    ):
        """
        return_all_blocks: if True, return the features at the end of every block
                           instead of just the features from the last block (eg for some prediction heads)

        masks1 can be None => assume image1 fully visible
        """
        # Encoder to decoder layer.
        visf1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)
        # Append masked tokens to the sequence.
        B, Nenc, C = visf1.size()
        if masks1 is None:  # downstreams
            f1_ = visf1
        else:  # pretraining
            Ntotal = masks1.size(1)
            f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
            f1_[~masks1] = visf1.view(B * Nenc, C)
        # Add positional embedding.
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
            f2 = f2 + self.dec_pos_embed
        # Apply Transformer blocks.
        out = f1_
        out2 = f2
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                _out, out2 = blk(_out, out2, pos1, pos2)
                out.append(_out)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out, out2 = blk(out, out2, pos1, pos2)
            out = self.dec_norm(out)
        return out

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        if not (imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0):
            raise ValueError(
                f"Image must be square and divisible by patch_size={p}, got"
                f" shape {imgs.shape}"
            )

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def forward(self, img1, img2):
        """
        img1: tensor of size B x 3 x img_size x img_size
        img2: tensor of size B x 3 x img_size x img_size

        out will be    B x N x (3*patch_size*patch_size)
        masks are also returned as B x N just in case
        """
        # Encoder of the masked first image.
        feat1, pos1, mask1 = self._encode_image(img1, do_mask=True)
        # Encoder of the second image.
        feat2, pos2, _ = self._encode_image(img2, do_mask=False)
        # Decoder.
        decfeat = self._decoder(feat1, pos1, mask1, feat2, pos2)
        # Prediction head.
        out = self.prediction_head(decfeat)
        # Get target.
        target = self.patchify(img1)
        return out, mask1, target


T = TypeVar("T")


class Backbone(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
    ) -> Float[Tensor, "batch view d_out height width"]:
        pass

    @property
    @abstractmethod
    def d_out(self) -> int:
        pass


inf = float("inf")


croco_params = {
    "ViTLarge_BaseDecoder": {
        "enc_depth": 24,
        "dec_depth": 12,
        "enc_embed_dim": 1024,
        "dec_embed_dim": 768,
        "enc_num_heads": 16,
        "dec_num_heads": 12,
        "pos_embed": "RoPE100",
        "img_size": (512, 512),
    },
}

default_dust3r_params = {
    "enc_depth": 24,
    "dec_depth": 12,
    "enc_embed_dim": 1024,
    "dec_embed_dim": 768,
    "enc_num_heads": 16,
    "dec_num_heads": 12,
    "pos_embed": "RoPE100",
    "patch_embed_cls": "PatchEmbedDust3R",
    "img_size": (512, 512),
    "head_type": "dpt",
    "output_mode": "pts3d",
    "depth_mode": ("exp", -inf, inf),
    "conf_mode": ("exp", 1, inf),
}


@dataclass
class BackboneCrocoCfg:
    name: Literal["croco", "croco_multi"]
    # Keep interface for the last two models, but they are not supported.
    model: Literal[
        "ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"
    ]
    # PatchEmbedDust3R or ManyAR_PatchEmbed.
    patch_embed_cls: str = "PatchEmbedDust3R"
    asymmetry_decoder: bool = True
    intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = "none"
    intrinsics_embed_degree: int = 0
    # Linear or dpt.
    intrinsics_embed_type: Literal["pixelwise", "linear", "token"] = "token"


BackboneCfg = BackboneCrocoCfg


class AsymmetricCroCo(CroCoNet):
    """
    Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:

        self.intrinsics_embed_loc = cfg.intrinsics_embed_loc
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        self.intrinsics_embed_encoder_dim = 0
        self.intrinsics_embed_decoder_dim = 0
        if (
            self.intrinsics_embed_loc == "encoder"
            and self.intrinsics_embed_type == "pixelwise"
        ):
            self.intrinsics_embed_encoder_dim = (
                (self.intrinsics_embed_degree + 1) ** 2
                if self.intrinsics_embed_degree > 0
                else 3
            )
        elif (
            self.intrinsics_embed_loc == "decoder"
            and self.intrinsics_embed_type == "pixelwise"
        ):
            self.intrinsics_embed_decoder_dim = (
                (self.intrinsics_embed_degree + 1) ** 2
                if self.intrinsics_embed_degree > 0
                else 3
            )

        self.patch_embed_cls = cfg.patch_embed_cls
        self.croco_args = fill_default_args(
            croco_params[cfg.model], CroCoNet.__init__
        )

        super().__init__(**croco_params[cfg.model])

        if cfg.asymmetry_decoder:
            self.dec_blocks2 = deepcopy(
                self.dec_blocks
            )  # This is used in DUSt3R and MASt3R

        if (
            self.intrinsics_embed_type == "linear"
            or self.intrinsics_embed_type == "token"
        ):
            self.intrinsic_encoder = nn.Linear(9, 1024)

        # self.set_freeze(freeze)

    def _set_patch_embed(
        self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3
    ):
        in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # Transfer from encoder to decoder.
        enc_embed_dim = enc_embed_dim + self.intrinsics_embed_decoder_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # Transformer for the decoder.
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        # Final norm layer.
        self.dec_norm = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # Duplicate all weights for the second decoder if not present.
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks2") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks2")] = value
        return super().load_state_dict(new_ckpt, **kw)

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def _encode_image(self, image, true_shape, intrinsics_embed=None):
        # Embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        if intrinsics_embed is not None:

            if self.intrinsics_embed_type == "linear":
                x = x + intrinsics_embed
            elif self.intrinsics_embed_type == "token":
                x = torch.cat((x, intrinsics_embed), dim=1)
                add_pose = pos[:, 0:1, :].clone()
                add_pose[:, :, 0] += pos[:, -1, 0].unsqueeze(-1) + 1
                pos = torch.cat((pos, add_pose), dim=1)

        # Add positional embedding without cls token.
        if self.enc_pos_embed is not None:
            raise ValueError("enc_pos_embed must be None")

        # Now apply the transformer encoder and normalization.
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(
        self,
        img1,
        img2,
        true_shape1,
        true_shape2,
        intrinsics_embed1=None,
        intrinsics_embed2=None,
    ):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(
                torch.cat((img1, img2), dim=0),
                torch.cat((true_shape1, true_shape2), dim=0),
                (
                    torch.cat((intrinsics_embed1, intrinsics_embed2), dim=0)
                    if intrinsics_embed1 is not None
                    else None
                ),
            )
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(
                img1, true_shape1, intrinsics_embed1
            )
            out2, pos2, _ = self._encode_image(
                img2, true_shape2, intrinsics_embed2
            )
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2, force_asym=False):
        img1 = view1["img"]
        img2 = view2["img"]
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one.
        shape1 = view1.get(
            "true_shape", torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        )
        shape2 = view2.get(
            "true_shape", torch.tensor(img2.shape[-2:])[None].repeat(B, 1)
        )
        # Warning! maybe the images have different portrait/landscape orientations.

        intrinsics_embed1 = view1.get("intrinsics_embed", None)
        intrinsics_embed2 = view2.get("intrinsics_embed", None)

        if force_asym or not is_symmetrized(view1, view2):
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                img1, img2, shape1, shape2, intrinsics_embed1, intrinsics_embed2
            )
        else:
            # Computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                img1[::2], img2[::2], shape1[::2], shape2[::2]
            )
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(
        self, f1, pos1, f2, pos2, extra_embed1=None, extra_embed2=None
    ):
        final_output = [(f1, f2)]  # before projection

        if extra_embed1 is not None:
            f1 = torch.cat((f1, extra_embed1), dim=-1)
        if extra_embed2 is not None:
            f2 = torch.cat((f2, extra_embed2), dim=-1)

        # Project to decoder dim.
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # Img1 side.
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # Img2 side.
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # Store the result.
            final_output.append((f1, f2))

        # Normalize last output.
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # Img_shape = tuple(map(int, img_shape))
        head = getattr(self, f"head{head_num}")
        return head(decout, img_shape)

    def forward(
        self,
        context: dict,
        symmetrize_batch=False,
        return_views=False,
    ):
        b, v, _, h, w = context["image"].shape
        device = context["image"].device

        view1, view2 = (
            {"img": context["image"][:, 0]},
            {"img": context["image"][:, 1]},
        )

        # Camera embedding in the encoder.
        if (
            self.intrinsics_embed_loc == "encoder"
            and self.intrinsics_embed_type == "pixelwise"
        ):
            intrinsic_emb = get_intrinsic_embedding(
                context, degree=self.intrinsics_embed_degree
            )
            view1["img"] = torch.cat((view1["img"], intrinsic_emb[:, 0]), dim=1)
            view2["img"] = torch.cat((view2["img"], intrinsic_emb[:, 1]), dim=1)

        if self.intrinsics_embed_loc == "encoder" and (
            self.intrinsics_embed_type == "token"
            or self.intrinsics_embed_type == "linear"
        ):
            intrinsic_embedding = self.intrinsic_encoder(
                context["intrinsics"].flatten(2)
            )
            view1["intrinsics_embed"] = intrinsic_embedding[:, 0].unsqueeze(1)
            view2["intrinsics_embed"] = intrinsic_embedding[:, 1].unsqueeze(1)

        if symmetrize_batch:
            instance_list_view1, instance_list_view2 = [0 for _ in range(b)], [
                1 for _ in range(b)
            ]
            view1["instance"] = instance_list_view1
            view2["instance"] = instance_list_view2
            view1["idx"] = instance_list_view1
            view2["idx"] = instance_list_view2
            view1, view2 = make_batch_symmetric(view1, view2)

            # Encode the two images --> B,S,D.
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = (
                self._encode_symmetrized(view1, view2, force_asym=False)
            )
        else:
            # Encode the two images --> B,S,D.
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = (
                self._encode_symmetrized(view1, view2, force_asym=True)
            )

        if self.intrinsics_embed_loc == "decoder":
            # FIXME: downsample is hardcoded to 16
            intrinsic_emb = get_intrinsic_embedding(
                context,
                degree=self.intrinsics_embed_degree,
                downsample=16,
                merge_hw=True,
            )
            dec1, dec2 = self._decoder(
                feat1,
                pos1,
                feat2,
                pos2,
                intrinsic_emb[:, 0],
                intrinsic_emb[:, 1],
            )
        else:
            dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        if (
            self.intrinsics_embed_loc == "encoder"
            and self.intrinsics_embed_type == "token"
        ):
            dec1, dec2 = list(dec1), list(dec2)
            for i in range(len(dec1)):
                dec1[i] = dec1[i][:, :-1]
                dec2[i] = dec2[i][:, :-1]

        if return_views:
            return dec1, dec2, shape1, shape2, view1, view2
        return dec1, dec2, shape1, shape2

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024


def get_backbone(cfg: BackboneCfg, d_in: int = 3) -> nn.Module:
    return AsymmetricCroCo(cfg, d_in)
