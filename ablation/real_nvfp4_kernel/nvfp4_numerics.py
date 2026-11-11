"""NVFP4 numerics: the `blockfmt` backend for this repo, built on torchao's NVFP4 recipe.

The repo resolves its block-format library from $BLOCKFMT_BACKEND (see blockfmt/__init__.py).
That library is not available on this machine, so this module provides a drop-in `Format`
with the same surface (`Format.from_shorthand("NVFP4[E2M1]{16}").cast(x, -1)`) whose numerics
follow torchao's `nvfp4_quantize` (torchao/prototype/mx_formats/nvfp4_tensor.py), i.e. the
same NVFP4 definition that torch._scaled_mm / cuBLAS consume on Blackwell:

    element  : FP4 E2M1, round-to-nearest-even, saturating at +-6
    block    : 16 contiguous elements along the last dim share one FP8 UE4M3 scale
    tensor   : (two-level mode) one FP32 per-tensor scale g = amax / (448 * 6), so the
               block scales use the full E4M3 range regardless of the tensor's magnitude

  export BLOCKFMT_BACKEND=nvfp4_numerics          # with this directory on PYTHONPATH

Only NVFP4 is implemented. Asking for any other shorthand raises, so this can never silently
stand in for an MX format.

Mode (env NVFP4_TENSOR_SCALE, default "single_level"):
  single_level : E4M3 block scales only (torchao's per_tensor_scale=None path). This is the
                 default because it is the one that reproduces the paper: fake-quant RTN W4A4
                 on Llama-3.1-8B gives WikiText-2 PPL 9.03 here vs 9.04 in Table 1, while
                 two_level gives a very different number (see logs/ and REPRODUCE.md).
  two_level    : dynamic per-tensor FP32 scale + E4M3 block scales (torchao default for
                 NVFP4DynamicActivationNVFP4WeightConfig(use_dynamic_per_tensor_scale=True))

The real kernel (nvfp4_kernels.py) implements the same arithmetic op-for-op, and
verification/test_act_quant_bitexact.py checks the two agree bit for bit.
"""
import os

import torch

F4_MAX = 6.0
E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)     # 448
E4M3_TINY = float(torch.finfo(torch.float8_e4m3fn).tiny)   # 2^-6, torchao's E4M3_EPS
BLOCK = 16
MODE = os.environ.get("NVFP4_TENSOR_SCALE", "single_level")
assert MODE in ("two_level", "single_level"), MODE

# E2M1 magnitudes indexed by their 3-bit code
FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fp4_rne_code(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> 4-bit E2M1 code (sign in bit 3), round-to-nearest-even, saturating.
    Tie handling: 0.25->0, 0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5->4 (even codes win)."""
    a = v.abs()
    m = torch.zeros_like(a, dtype=torch.uint8)
    m = torch.where(a > 0.25, 1, m)
    m = torch.where(a >= 0.75, 2, m)
    m = torch.where(a > 1.25, 3, m)
    m = torch.where(a >= 1.75, 4, m)
    m = torch.where(a > 2.5, 5, m)
    m = torch.where(a >= 3.5, 6, m)
    m = torch.where(a > 5.0, 7, m).to(torch.uint8)
    return m | ((v < 0).to(torch.uint8) << 3)


_LUT = None


def fp4_code_to_f32(code: torch.Tensor) -> torch.Tensor:
    global _LUT
    if _LUT is None or _LUT.device != code.device:
        lv = torch.tensor(FP4_LEVELS, dtype=torch.float32, device=code.device)
        _LUT = torch.cat([lv, -lv])
    return _LUT[code.long()]


def tensor_scale(amax: torch.Tensor) -> torch.Tensor:
    """torchao per_tensor_amax_to_scale; guarded so an all-zero tensor gives g=1, not 0/NaN."""
    g = amax.to(torch.float32) / (E4M3_MAX * F4_MAX)
    return torch.where(g > 0, g, torch.ones_like(g))


def nvfp4_quantize_ref(x: torch.Tensor, global_scale="auto", batch_dims=0):
    """Reference NVFP4 quantizer. Returns (codes uint8 [...,K], block scales e4m3 [...,K/16],
    global scale fp32 or None). Mirrors torchao.nvfp4_quantize op-for-op.

    global_scale: "auto" -> derived from this tensor's amax according to MODE;
                  None   -> single-level; a tensor -> use it.
    batch_dims:   leading dims that are independent tensors (each gets its own amax).
    """
    shp = x.shape
    K = shp[-1]
    assert K % BLOCK == 0, f"last dim {K} not divisible by {BLOCK}"
    xf = x.to(torch.float32).reshape(*shp[:-1], K // BLOCK, BLOCK)
    bmax = xf.abs().amax(-1)                                   # [..., K/16]
    if isinstance(global_scale, str):
        if MODE == "single_level":
            global_scale = None
        else:
            red = tuple(range(batch_dims, bmax.dim()))
            amax = bmax.amax(dim=red, keepdim=True) if red else bmax
            global_scale = tensor_scale(amax)
    bs = bmax / F4_MAX
    if global_scale is None:
        s8 = bs.clamp(min=E4M3_TINY, max=E4M3_MAX).to(torch.float8_e4m3fn)
        recip = 1.0 / s8.to(torch.float32)
    else:
        s8 = (bs / global_scale).clamp(min=E4M3_TINY, max=E4M3_MAX).to(torch.float8_e4m3fn)
        recip = (1.0 / global_scale) / s8.to(torch.float32)
    v = (xf * recip.unsqueeze(-1)).clamp(-F4_MAX, F4_MAX)
    codes = fp4_rne_code(v).reshape(shp)
    return codes, s8, global_scale


def nvfp4_dequant_ref(codes, s8, global_scale):
    shp = codes.shape
    vals = fp4_code_to_f32(codes).reshape(*shp[:-1], shp[-1] // BLOCK, BLOCK)
    sc = s8.to(torch.float32)
    if global_scale is not None:
        sc = sc * global_scale
    return (vals * sc.unsqueeze(-1)).reshape(shp)


def nvfp4_fake_quant(x: torch.Tensor, batch_dims=0) -> torch.Tensor:
    codes, s8, g = nvfp4_quantize_ref(x, "auto", batch_dims=batch_dims)
    return nvfp4_dequant_ref(codes, s8, g).to(x.dtype)


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """[...,K] uint8 codes -> [...,K/2] bytes; element 2i in the low nibble (torchao order)."""
    c = codes.contiguous()
    return c[..., 0::2] | (c[..., 1::2] << 4)


def to_blocked_scales(s: torch.Tensor) -> torch.Tensor:
    """[R, C] block scales -> cuBLAS 32_4_4 swizzled 1-D layout (same as torchao to_blocked)."""
    R, C = s.shape
    nrb, ncb = -(-R // 128), -(-C // 4)
    p = torch.zeros(nrb * 128, ncb * 4, dtype=s.dtype, device=s.device)
    p[:R, :C] = s
    return (p.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
             .reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1))


class _NVFP4Format:
    block_size = BLOCK

    def cast(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim not in (-1, x.dim() - 1):
            raise NotImplementedError("NVFP4 cast only along the last dim")
        return nvfp4_fake_quant(x)


class Format:
    @staticmethod
    def from_shorthand(s: str):
        if s.replace(" ", "") == "NVFP4[E2M1]{16}":
            return _NVFP4Format()
        raise NotImplementedError(
            f"nvfp4_numerics only implements NVFP4[E2M1]{{16}}, not {s!r}")
