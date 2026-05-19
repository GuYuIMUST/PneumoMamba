# --------------------------------------------------------
# Omnidirectional Cross-Scan Operations for State Space Model
# Modified by Abdelrahman Shaker
# --------------------------------------------------------
# VMamba: Visual State Space Model
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch


# Original 4-directional cross scan =============
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


# Partial scan direction 1: left-to-right row scan
class CrossScan_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        y = ys[:, 0]
        return y.view(B, -1, H, W)


class CrossMerge_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0]
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = x
        xs = xs.view(B, 1, C, H, W)
        return xs


# Partial scan direction 2: top-to-bottom column scan
class CrossScan_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C, H * W))
        xs[:, 0] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0].view(B, -1, H, W).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs = xs.view(B, 1, C, H, W)
        return xs


# Partial scan direction 3: right-to-left row scan (reversed)
class CrossScan_3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C, H * W))
        xs[:, 0] = torch.flip(x.flatten(2, 3), dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0].flip(dims=[-1]).view(B, 1, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge_3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0].flip(dims=[-1]).view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = torch.flip(x, dims=[-1])
        xs = xs.view(B, 1, C, H, W)
        return xs


# Partial scan direction 4: bottom-to-top column scan (reversed)
class CrossScan_4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C, H * W))
        xs[:, 0] = torch.flip(x.transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0].view(B, -1, H, W).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L).flip(dims=[-1]).view(B, 1, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge_4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1).flip(dims=[-1])
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = torch.flip(x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1])
        xs = xs.view(B, 1, C, H, W)
        return xs


# Left-right flipped scan
class CrossScan_LR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x_flipped = torch.flip(x, [3])  # Flip along width dimension
        xs = x_flipped.new_empty((B, 1, C, H * W))
        xs[:, 0] = x_flipped.flatten(2, 3)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        y = ys[:, 0]
        return y.view(B, -1, H, W)


class CrossMerge_LR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys_flipped = torch.flip(ys, [3])  # Flip along width dimension
        ys = ys_flipped.view(B, K, D, -1)
        y = ys[:, 0]
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = x
        xs = xs.view(B, 1, C, H, W)
        return xs


# Up-down flipped scan
class CrossScan_UD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x_flipped = torch.flip(x, [2])  # Flip along height dimension
        xs = x_flipped.new_empty((B, 1, C, H * W))
        xs[:, 0] = x_flipped.flatten(2, 3)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        y = ys[:, 0]
        return y.view(B, -1, H, W)


class CrossMerge_UD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys_flipped = torch.flip(ys, [2])  # Flip along height dimension
        ys = ys_flipped.view(B, K, D, -1)
        y = ys[:, 0]
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = x
        xs = xs.view(B, 1, C, H, W)
        return xs


# Transposed then up-down flipped scan
class CrossScan_T_UD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x_transposed = x.permute(0, 2, 3, 1)
        x_flipped = torch.flip(x_transposed, [1])  # Flip along height dimension
        xs = x_flipped.new_empty((B, 1, C, H * W))
        xs[:, 0] = x_flipped.flatten(2, 3)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        y = ys[:, 0]
        return y.view(B, -1, H, W)


class CrossMerge_T_UD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys_transposed = ys.permute(0, 1, 3, 4, 2)
        ys_flipped = torch.flip(ys_transposed, [2])  # Flip along height dimension
        ys = ys_flipped.view(B, K, D, -1)
        y = ys[:, 0]
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = x
        xs = xs.view(B, 1, C, H, W)
        return xs


# Transposed then left-right flipped scan
class CrossScan_T_LR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x_transposed = x.permute(0, 2, 3, 1)
        x_flipped = torch.flip(x_transposed, [3])  # Flip along width dimension
        xs = x_flipped.new_empty((B, 1, C, H * W))
        xs[:, 0] = x_flipped.flatten(2, 3)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        y = ys[:, 0]
        return y.view(B, -1, H, W)


class CrossMerge_T_LR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys_transposed = ys.permute(0, 1, 3, 4, 2)
        ys_flipped = torch.flip(ys_transposed, [3])  # Flip along width dimension
        ys = ys_flipped.view(B, K, D, -1)
        y = ys[:, 0]
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = x
        xs = xs.view(B, 1, C, H, W)
        return xs


# Import selective scan CUDA kernels ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...

try:
    import selective_scan_cuda_core
except Exception as e:
    ...

try:
    import selective_scan_cuda
except Exception as e:
    ...


def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb;
            pdb.set_trace()


# FLOPs calculation =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    assert not with_complex
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
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


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# Selective scan CUDA wrappers ===============================
class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus, False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops
