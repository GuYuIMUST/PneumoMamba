"""
PneumonMamba: A novel Mamba-based Network for Computer-Aided Diagnosis of Pneumonia
with Omnidirectional Scanning and Multi-Scale Convolution.

Key innovations:
1. Omnidirectional State Space Model (OSSM) with 8-directional scanning
2. Multi-scale Dilated Residual Module (MDR) for multi-scale feature extraction
3. Dual Attention Module (DAM) combining channel and spatial attention
4. Channel Affinity-based Group Mamba Layer with SE-style channel modulation
"""

import time
import math
from functools import partial
from typing import Optional, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from dual_attention import DualAttentionModule
from multi_scale_conv import MultiScaleConvModule
from wtconv2d import WTConv2d

try:
    from .csms6s_plus import (
        CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4,
        CrossScan_LR, CrossScan_UD, CrossScan_T_UD, CrossScan_T_LR,
        CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4,
        CrossMerge_LR, CrossMerge_UD, CrossMerge_T_UD, CrossMerge_T_LR
    )
except ImportError:
    from csms6s_plus import (
        CrossScan_1, CrossScan_2, CrossScan_3, CrossScan_4,
        CrossScan_LR, CrossScan_UD, CrossScan_T_UD, CrossScan_T_LR,
        CrossMerge_1, CrossMerge_2, CrossMerge_3, CrossMerge_4,
        CrossMerge_LR, CrossMerge_UD, CrossMerge_T_UD, CrossMerge_T_LR
    )

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except ImportError:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """Compute FLOPs for selective scan reference implementation."""
    import numpy as np

    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex
    flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L

    return flops


# ===================== Patch Embedding & Merging =====================

class PatchEmbed2D(nn.Module):
    """Image to Patch Embedding via convolution projection.

    Args:
        patch_size: Size of each patch (default: 4).
        in_chans: Number of input image channels (default: 3).
        embed_dim: Output embedding dimension (default: 96).
        norm_layer: Normalization layer (default: None).
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    """Patch Merging Layer for downsampling feature maps by 2x.

    Args:
        dim: Number of input channels.
        norm_layer: Normalization layer (default: nn.LayerNorm).
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        # Split input into 4 patches of H/2 x W/2
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    """Patch Expansion Layer for upsampling feature maps by 2x."""

    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        return x


class Final_PatchExpand2D(nn.Module):
    """Final Patch Expansion Layer for upsampling to target resolution."""

    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        return x


# ===================== Group Mamba Layer (8-direction OSSM) =====================

class GroupMambaLayer(nn.Module):
    """Channel Affinity-based Group Mamba Layer with 8-directional scanning.

    Splits input channels into 8 groups and applies independent SS2D blocks
    with different scanning directions. Includes SE-style channel modulation.

    Args:
        input_dim: Input channel dimension.
        output_dim: Output channel dimension.
        d_state: SSM state dimension.
        d_conv: Convolution kernel size in SSM.
        expand: SSM expansion ratio.
        reduction: Channel reduction ratio for SE attention.
    """

    def __init__(self, input_dim, output_dim, d_state=1, d_conv=3, expand=1, reduction=16):
        super().__init__()

        num_channels_reduced = input_dim // reduction
        self.fc1 = nn.Linear(input_dim, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, output_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        # 8 independent SS2D blocks for 8 scanning directions
        self.mamba_g1 = SS2D(d_model=input_dim // 8, d_state=d_state, ssm_ratio=expand, d_conv=d_conv)
        self.mamba_g2 = SS2D(d_model=input_dim // 8, d_state=d_state, ssm_ratio=expand, d_conv=d_conv)
        self.mamba_g3 = SS2D(d_model=input_dim // 8, d_state=d_state, ssm_ratio=expand, d_conv=d_conv)
        self.mamba_g4 = SS2D(d_model=input_dim // 8, d_state=d_state, ssm_ratio=expand, d_conv=d_conv)
        self.mamba_g5 = SS2D(d_model=input_dim // 8, d_state=d_state, ssm_ratio=expand, d_conv=d_conv)
        self.mamba_g6 = SS2D(d_model=input_dim // 8, d_state=d_state, ssm_ratio=expand, d_conv=d_conv)
        self.mamba_g7 = SS2D(d_model=input_dim // 8, d_state=d_state, ssm_ratio=expand, d_conv=d_conv)
        self.mamba_g8 = SS2D(d_model=input_dim // 8, d_state=d_state, ssm_ratio=expand, d_conv=d_conv)

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, H, W):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        x = x.reshape(B, N, C)  # (B, N, C)

        B, N, C = x.shape
        x = self.norm(x)

        # Channel Affinity: compute channel-wise attention weights
        z = x.permute(0, 2, 1).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(z))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        x = rearrange(x, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)

        # Split input into 8 channel groups
        x1, x2, x3, x4, x5, x6, x7, x8 = torch.chunk(x, 8, dim=-1)

        # Apply 8 different scanning directions to each group
        x_mamba1 = self.mamba_g1(x1, CrossScan=CrossScan_1, CrossMerge=CrossMerge_1)
        x_mamba2 = self.mamba_g2(x2, CrossScan=CrossScan_2, CrossMerge=CrossMerge_2)
        x_mamba3 = self.mamba_g3(x3, CrossScan=CrossScan_3, CrossMerge=CrossMerge_3)
        x_mamba4 = self.mamba_g4(x4, CrossScan=CrossScan_4, CrossMerge=CrossMerge_4)
        x_mamba5 = self.mamba_g5(x5, CrossScan=CrossScan_LR, CrossMerge=CrossScan_LR)
        x_mamba6 = self.mamba_g6(x6, CrossScan=CrossScan_UD, CrossMerge=CrossMerge_UD)
        x_mamba7 = self.mamba_g7(x7, CrossScan=CrossScan_T_LR, CrossMerge=CrossMerge_T_LR)
        x_mamba8 = self.mamba_g8(x8, CrossScan=CrossScan_T_UD, CrossMerge=CrossMerge_T_UD)

        # Combine all feature maps with residual scaling
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4,
                             x_mamba5, x_mamba6, x_mamba7, x_mamba8], dim=-1) * self.skip_scale * x

        x_mamba = rearrange(x_mamba, 'b h w c -> b (h w) c', b=B, h=H, w=W, c=C)

        # Channel Modulation
        x_mamba = x_mamba * fc_out_2.unsqueeze(1)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)

        return x_mamba


# ===================== State Space 2D Block (SS2D) =====================

class SS2D(nn.Module):
    """2D Selective State Space block with Dual Attention and Multi-scale Convolution.

    Integrates the Mamba selective scan mechanism with DAM (Dual Attention Module)
    and MDR (Multi-scale Dilated Residual) module for enhanced feature extraction.
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Dual Attention Module (DAM)
        self.dual_att = DualAttentionModule(in_channels=self.d_inner, out_channels=self.d_inner)
        # Multi-scale Dilated Residual Module (MDR)
        self.conv = MultiScaleConvModule(in_channels=self.d_inner, out_channels=self.d_model)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        residual = x
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        # Apply Dual Attention Module on gating branch
        z = z.permute(0, 3, 1, 2).contiguous()  # B C H W
        z = self.dual_att(z)  # B C H W
        z = z.permute(0, 2, 3, 1).contiguous()
        y = y * F.silu(z)
        out = self.out_proj(y)

        # Apply Multi-scale Dilated Residual module
        out = y.permute(0, 3, 1, 2).contiguous()  # B C H W
        out = self.conv(out)
        out = out.permute(0, 2, 3, 1).contiguous()  # B H W C

        if self.dropout is not None:
            out = self.dropout(out)

        return out + residual


# ===================== Conv & Attention Blocks =====================

class ConvBlock(nn.Module):
    """Bottleneck-style convolutional block with optional residual connection."""

    def __init__(self, in_channels, out_channels, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        expansion = 4
        c = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(c)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(c)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_bn = norm_layer(out_channels)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class Encoding(nn.Module):
    """Feature encoding layer with learnable codewords for dictionary learning."""

    def __init__(self, in_channels, num_codes):
        super(Encoding, self).__init__()
        self.in_channels, self.num_codes = in_channels, num_codes
        num_codes = 64
        std = 1. / ((num_codes * in_channels) ** 0.5)
        self.codewords = nn.Parameter(
            torch.empty(num_codes, in_channels, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, in_channels = codewords.size()
        b = x.size(0)
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))
        reshaped_scale = scale.view((1, 1, num_codes))
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, in_channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))
        b = x.size(0)
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))
        assignment_weights = assignment_weights.unsqueeze(3)
        encoded_feat = (assignment_weights * (expanded_x - reshaped_codewords)).sum(1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        b, in_channels, w, h = x.size()
        x = x.view(b, self.in_channels, -1).transpose(1, 2).contiguous()
        assignment_weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """Conv2d -> BatchNorm -> Activation block."""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride,
                              padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class MSConv(nn.Module):
    """Multi-scale depthwise convolution with multiple kernel sizes."""

    def __init__(self, dim: int, kernel_sizes: Sequence[int] = (1, 3, 5)) -> None:
        super(MSConv, self).__init__()
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
            for kernel_size in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + sum([conv(x) for conv in self.dw_convs])


class DWConv(nn.Module):
    """Depthwise Separable Convolution."""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class LVCBlock(nn.Module):
    """Lightweight Visual Coding block with dictionary encoding and SE attention."""

    def __init__(self, in_channels, out_channels, num_codes, channel_ratio=0.25, base_channel=64):
        super(LVCBlock, self).__init__()
        self.out_channels = out_channels
        self.num_codes = num_codes
        num_codes = 64

        self.conv_1 = ConvBlock(in_channels=in_channels, out_channels=in_channels, res_conv=True, stride=1)

        self.LVC = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Encoding(in_channels=in_channels, num_codes=num_codes),
            nn.BatchNorm1d(num_codes),
            nn.ReLU(inplace=True),
            Mean(dim=1))

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())

    def forward(self, x):
        x = self.conv_1(x, return_x_2=False)
        en = self.LVC(x)
        gam = self.fc(en)
        b, in_channels, _, _ = x.size()
        y = gam.view(b, in_channels, 1, 1)
        x = F.relu_(x + x * y)
        return x


class Mlp(nn.Module):
    """MLP with 1x1 convolutions. Input: tensor with shape [B, C, H, W]."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupNorm(nn.GroupNorm):
    """Group Normalization with 1 group. Input: tensor in shape [B, C, H, W]."""

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """Channel shuffle operation for mixing information across groups."""
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, height, width, groups, channels_per_group)
    x = torch.transpose(x, 3, 4).contiguous()
    x = x.view(batch_size, height, width, -1)
    return x


class LightMLPBlock(nn.Module):
    """Lightweight MLP block with depthwise convolution and layer scale."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu",
                 mlp_ratio=4., drop=0., act_layer=nn.GELU,
                 use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0., norm_layer=GroupNorm):
        super().__init__()
        self.dw = DWConv(in_channels, out_channels, ksize=1, stride=1, act="silu")
        self.linear = nn.Linear(out_channels, out_channels)
        self.out_channels = out_channels

        self.norm1 = norm_layer(in_channels)
        self.norm2 = norm_layer(in_channels)

        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(in_features=in_channels, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dw(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.dw(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EVCBlock(nn.Module):
    """Enhanced Visual Coding block combining LVC and LightMLP branches."""

    def __init__(self, in_channels, out_channels, channel_ratio=4, base_channel=16):
        super().__init__()
        expansion = 2
        ch = in_channels * expansion

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.lvc = LVCBlock(in_channels=in_channels, out_channels=in_channels, num_codes=64)
        self.l_MLP = LightMLPBlock(in_channels, in_channels, ksize=1, stride=1, act="silu", act_layer=nn.GELU,
                                   mlp_ratio=4., drop=0., use_layer_scale=True, layer_scale_init_value=1e-5,
                                   drop_path=0., norm_layer=GroupNorm)

        self.cnv1 = nn.Conv2d(ch, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        x_lvc = self.lvc(x1)
        x_lmlp = self.l_MLP(x1)
        x = torch.cat((x_lvc, x_lmlp), dim=1)
        x = self.cnv1(x)
        return x


# ===================== Pooling & Attention Modules =====================

class AvgPoolingChannel(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class MaxPoolingChannel(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)


class SEAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channel=3, reduction=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class QSEME(nn.Module):
    """Quad-Stream Enhanced Multi-scale Extraction module.

    Splits input into 4 branches: max pooling, avg pooling, wavelet convolution, and identity.
    """

    def __init__(self, out_c):
        super().__init__()
        self.out_c = out_c
        self.se = SEAttention(channel=out_c)
        self.maxpool = MaxPoolingChannel()
        self.avgpool = AvgPoolingChannel()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )

        self.wtconv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )
        self.wtconv2 = nn.Sequential(
            WTConv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )
        self.wtconv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.out_c, kernel_size=1),
            nn.BatchNorm2d(num_features=self.out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.init_conv(x)
        x1, x2, x3, x4 = x.chunk(4, dim=1)  # Split into 4 branches, each 8 channels

        # Branch 1: max pooling with upsampling
        channel_1_max_pool = self.maxpool(x1)
        desired_size = (x1.size(2), x1.size(3))
        channel_1_max_pool_out = F.interpolate(channel_1_max_pool, size=desired_size, mode='bilinear', align_corners=False)

        # Branch 2: average pooling with upsampling
        channel_2_avg_pool = self.avgpool(x2)
        desired_size = (x2.size(2), x2.size(3))
        channel_2_avg_pool_out = F.interpolate(channel_2_avg_pool, size=desired_size, mode='bilinear', align_corners=False)

        # Branch 3: wavelet convolution
        channel_3_1 = self.wtconv1(x3)
        channel_3_2 = self.wtconv2(channel_3_1)
        channel_3_3_out = self.wtconv3(channel_3_2)

        # Branch 4: identity (residual)
        channel_4 = x4

        output = torch.cat((channel_1_max_pool_out, channel_2_avg_pool_out, channel_3_3_out, channel_4), dim=1)
        output = self.out_conv(output)
        output = self.se(output)
        return output


# ===================== VSS Layers =====================

class SS_Conv_SSM(nn.Module):
    """Hybrid block combining convolutional branch and SSM (Mamba) branch with channel shuffle."""

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim // 2)
        self.self_attention = GroupMambaLayer(hidden_dim // 2, hidden_dim // 2)
        self.evc_block = EVCBlock(in_channels=hidden_dim, out_channels=hidden_dim)
        self.drop_path = DropPath(drop_path)

        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 2),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, input: torch.Tensor):
        input_left, input_right = input.chunk(2, dim=-1)
        H, W = input_right.shape[1], input_right.shape[2]

        # SSM (Mamba) branch
        x = self.drop_path(self.self_attention(self.ln_1(input_right), H, W))

        # Convolutional branch
        input_left = input_left.permute(0, 3, 1, 2).contiguous()
        input_left = self.conv33conv33conv11(input_left)
        input_left = input_left.permute(0, 2, 3, 1).contiguous()

        B, H, W, C = input_left.shape
        x = x.reshape(B, H, W, C)

        # Concatenate and shuffle channels
        output = torch.cat((input_left, x), dim=-1)
        output = channel_shuffle(output, groups=2)
        return output


class VSSLayer(nn.Module):
    """Visual State Space layer with optional downsampling.

    Args:
        dim: Number of input channels.
        depth: Number of SS_Conv_SSM blocks.
        d_state: SSM state dimension.
        drop_path: Stochastic depth rate.
        norm_layer: Normalization layer.
        downsample: Downsample layer at the end.
        use_checkpoint: Whether to use gradient checkpointing.
    """

    def __init__(self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SS_Conv_SSM(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSLayer_up(nn.Module):
    """Visual State Space layer with optional upsampling."""

    def __init__(self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 upsample=None, use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SS_Conv_SSM(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


# ===================== Main Model: VSSM (PneumonMamba) =====================

class VSSM(nn.Module):
    """PneumonMamba: Visual State Space Model for pneumonia classification.

    A hierarchical encoder using Mamba-based state space blocks with
    omnidirectional scanning for medical image classification.

    Args:
        patch_size: Patch embedding size (default: 4).
        in_chans: Input image channels (default: 3).
        num_classes: Number of classification categories.
        depths: Number of blocks at each stage.
        dims: Channel dimensions at each stage.
        d_state: SSM state dimension.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalization layer.
        patch_norm: Whether to apply normalization after patch embedding.
        use_checkpoint: Whether to use gradient checkpointing.
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 4, 2],
                 depths_decoder=[2, 9, 2, 2], dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96],
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]

        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        # QSEME preprocessing module
        self.qseme = QSEME(out_c=3)

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x


if __name__ == "__main__":
    medmamba_t = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=4).to("cuda")
    medmamba_s = VSSM(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=4).to("cuda")
    medmamba_b = VSSM(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024], num_classes=4).to("cuda")

    data = torch.randn(1, 3, 224, 224).to("cuda")
    print(medmamba_t(data).shape)
