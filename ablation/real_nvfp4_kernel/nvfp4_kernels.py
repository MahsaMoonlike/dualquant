"""Real packed NVFP4 W4A4 linear for DualQuant on Blackwell (sm_100 / sm_120).

  weights     : packed FP4 E2M1 (2 per byte) + FP8 E4M3 scale per 16 elements, stored in the
                cuBLAS 32_4_4 swizzled layout, + one FP32 per-tensor scale. Quantized once,
                offline, from DualQuant's mat_q. Never dequantized at runtime.
  activations : quantized on the fly to the same format by ONE Triton kernel that also applies
                DualQuant's column scale (x * col) while loading x, so beta costs no extra
                launch or extra pass over x. Scales are written directly in swizzled layout.
                Two-level mode needs the activation's per-tensor amax first; that is a small
                reduction kernel (_act_amax_kernel) which also applies col, so the activation
                path is amax + quantize (2 launches), single-level is quantize only (1 launch).
  GEMM        : cuBLASLt block-scaled FP4 GEMM via torch.nn.functional.scaled_mm, FP32
                accumulate, BF16 out. The two per-tensor scales are passed as TensorWise
                scales; torch multiplies them in a 1-element kernel (~1 us, counted under
                "other" in profile_breakdown) and hands the product to cuBLAS as alpha, so the
                output itself is never rescaled. A blockwise-only A operand (which would skip
                that kernel in single-level mode) is rejected by scaled_mm.

The activation kernel reproduces nvfp4_numerics.nvfp4_quantize_ref (= torchao.nvfp4_quantize)
op for op, with IEEE-rounded divisions, so the packed bytes and scales are bit-identical to
what the fake-quant path computes (verification/test_act_quant_bitexact.py).

Everything is torch-allocator / torch-stream based, so the whole model CUDA-graph captures
with no special handling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.functional import ScalingType, SwizzleType

import nvfp4_numerics as N

_FP4X2 = torch.float4_e2m1fn_x2
_REC = [ScalingType.BlockWise1x16, ScalingType.TensorWise]
_SWZ = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE]


# fp32(1/6) and fp32(1/2688), computed exactly the way torch's scalar division does
_INV6 = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(6.0, dtype=torch.float32))
_INV2688 = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(2688.0, dtype=torch.float32))


def swizzled_scale_numel(rows, kblocks):
    return triton.cdiv(rows, 128) * 128 * triton.cdiv(kblocks, 4) * 4


@triton.jit
def _act_amax_kernel(x_ptr, col_ptr, part_ptr, M, K, stride_xm,
                     BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pm = tl.program_id(0)
    pk = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pk * BLOCK_K + tl.arange(0, BLOCK_K)
    msk = (rm[:, None] < M) & (rk[None, :] < K)
    x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :], mask=msk, other=0.0).to(tl.float32)
    c = tl.load(col_ptr + rk, mask=rk < K, other=0.0)
    a = tl.max(tl.max(tl.abs(x * c[None, :]), axis=1), axis=0)
    tl.store(part_ptr + pm * tl.num_programs(1) + pk, a)


@triton.jit
def dq_nvfp4_quantize_act_kernel(
        x_ptr, col_ptr, part_ptr, n_part, q_ptr, s_ptr, g_ptr,
        M, K, stride_xm, n_cb, inv6, inv2688,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, NP: tl.constexpr,
        TWO_LEVEL: tl.constexpr, HAS_COL: tl.constexpr):
    pm = tl.program_id(0)
    pk = tl.program_id(1)

    if TWO_LEVEL:
        # every program reduces the (few) amax partials itself -> no extra launch, no atomics
        po = tl.arange(0, NP)
        amax = tl.max(tl.load(part_ptr + po, mask=po < n_part, other=0.0), axis=0)
        # torch evaluates `t / python_float` as t * fp32(1/float); mirror that bit for bit
        g = amax * inv2688                                # amax / (448 * 6)
        g = tl.where(g > 0.0, g, 1.0)
        inv_g = tl.div_rn(1.0, g)
        if (pm == 0) & (pk == 0):
            tl.store(g_ptr, g)

    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pk * BLOCK_K + tl.arange(0, BLOCK_K)
    msk = (rm[:, None] < M) & (rk[None, :] < K)
    x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :], mask=msk, other=0.0).to(tl.float32)
    if HAS_COL:
        c = tl.load(col_ptr + rk, mask=rk < K, other=0.0)
        x = x * c[None, :]                                # DualQuant beta, fused into the load

    xb = tl.reshape(x, (BLOCK_M, BLOCK_K // 16, 16))
    bmax = tl.max(tl.abs(xb), axis=2)                     # [BM, BK/16]
    bs = bmax * inv6                                      # bmax / 6, torch-style
    if TWO_LEVEL:
        bs = tl.div_rn(bs, g)
    bs = tl.minimum(tl.maximum(bs, 0.015625), 448.0)      # [E4M3 tiny, E4M3 max]
    s8 = bs.to(tl.float8e4nv)
    sf = s8.to(tl.float32)
    if TWO_LEVEL:
        rcp = tl.div_rn(inv_g, sf)
    else:
        rcp = tl.div_rn(1.0, sf)
    v = xb * rcp[:, :, None]
    v = tl.minimum(tl.maximum(v, -6.0), 6.0)

    # FP4 E2M1 round-to-nearest-even (same thresholds as nvfp4_numerics.fp4_rne_code)
    a = tl.abs(v)
    m = tl.where(a > 0.25, 1, 0)
    m = tl.where(a >= 0.75, 2, m)
    m = tl.where(a > 1.25, 3, m)
    m = tl.where(a >= 1.75, 4, m)
    m = tl.where(a > 2.5, 5, m)
    m = tl.where(a >= 3.5, 6, m)
    m = tl.where(a > 5.0, 7, m)
    code = (m | tl.where(v < 0.0, 8, 0)).to(tl.uint8)
    code = tl.reshape(code, (BLOCK_M, BLOCK_K // 2, 2))
    lo, hi = tl.split(code)
    packed = lo | (hi << 4)

    rk2 = pk * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(q_ptr + rm[:, None] * (K // 2) + rk2[None, :], packed,
             mask=(rm[:, None] < M) & (rk2[None, :] < K // 2))

    # scales, written straight into cuBLAS's 32_4_4 swizzled layout
    sc = pk * (BLOCK_K // 16) + tl.arange(0, BLOCK_K // 16)
    off = ((rm[:, None] // 128) * n_cb + sc[None, :] // 4) * 512 \
        + (rm[:, None] % 32) * 16 + ((rm[:, None] % 128) // 32) * 4 + sc[None, :] % 4
    tl.store(s_ptr + off, s8.to(tl.uint8, bitcast=True),
             mask=(rm[:, None] < M) & (sc[None, :] < K // 16))


def _tiles(M):
    return (16 if M <= 16 else 32), 128


_ONE = {}


def _unit_scale(dev):
    """Constant 1.0 tensor scale (single-level activations), allocated once per device so no
    fill kernel is launched per call."""
    if dev not in _ONE:
        _ONE[dev] = torch.ones((), dtype=torch.float32, device=dev)
    return _ONE[dev]


def quantize_act(x2d, col, two_level=None):
    """x2d [M,K] bf16 contiguous, col [K] fp32 or None -> (packed [M,K/2] u8, swizzled scales u8,
    g [] fp32). col=None (SmoothQuant: its scale is folded into the RMSNorm) compiles a kernel
    variant without the per-channel multiply."""
    two_level = (N.MODE == "two_level") if two_level is None else two_level
    M, K = x2d.shape
    BM, BK = _tiles(M)
    grid = (triton.cdiv(M, BM), triton.cdiv(K, BK))
    dev = x2d.device
    q = torch.empty((M, K // 2), dtype=torch.uint8, device=dev)
    s = torch.empty(swizzled_scale_numel(M, K // 16), dtype=torch.uint8, device=dev)
    g = torch.empty((), dtype=torch.float32, device=dev) if two_level else _unit_scale(dev)
    n_part = grid[0] * grid[1]
    part = torch.empty(n_part, dtype=torch.float32, device=dev) if two_level else g
    if two_level:
        assert col is not None, "two-level activations are only wired up with a column scale"
        _act_amax_kernel[grid](x2d, col, part, M, K, x2d.stride(0), BLOCK_M=BM, BLOCK_K=BK,
                               num_warps=4)
    dq_nvfp4_quantize_act_kernel[grid](
        x2d, x2d if col is None else col, part, n_part, q, s, g, M, K, x2d.stride(0),
        triton.cdiv(K // 16, 4),
        _INV6, _INV2688,
        BLOCK_M=BM, BLOCK_K=BK, NP=max(16, triton.next_power_of_2(n_part)),
        TWO_LEVEL=two_level, HAS_COL=col is not None, num_warps=4)
    return q, s, g


def quantize_weight(w, search=2):
    """Offline: [N,K] DualQuant mat_q -> (packed [N,K/2] u8, swizzled scales u8, g_w [] fp32).

    Why two-level, and why not just absmax: DualQuant's NVFP4 weight block (r, b) is
    alpha_inv[r,b] * s8[r,b] * fp4, i.e. TWO stacked E4M3 scales (the row scale and the inner
    block-format scale). The hardware has ONE E4M3 scale per block (+ one FP32 per tensor),
    and the product of two E4M3 values is generally not E4M3, so packing is not lossless in
    general. Two things keep the loss small, both offline and free at runtime:
      * g_w is a power of two, so every block whose stacked scale IS representable (e.g.
        alpha_inv a power of two, which the E4M3-tiny clamp makes common) packs exactly;
      * each block's E4M3 scale is the MSE-best of absmax/6, its +-`search` E4M3 neighbours,
        and absmax/{4,3,2} (ties keep absmax/6). The latter make blocks whose largest code is
        not 6 -- e.g. RTN blocks whose scale was clamped to E4M3 tiny -- pack exactly.
    The per-tensor scale is a per-layer constant folded into the GEMM alpha."""
    wf = w.to(torch.float32)
    R, C = wf.shape
    g = N.tensor_scale(wf.abs().amax())
    g = torch.exp2(torch.ceil(torch.log2(g)))           # power of two, block scales stay <= 448
    wb = wf.view(R, C // 16, 16)
    bmax = wb.abs().amax(-1)

    def e4m3_bits(v):
        return v.clamp(N.E4M3_TINY, N.E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int16)

    # candidate block scales: absmax/6 and its +-search E4M3 neighbours, plus absmax/{4,3,2}
    # (exact for blocks whose largest FP4 code is 4/3/2, e.g. scales clamped at E4M3 tiny)
    base = e4m3_bits(bmax / N.F4_MAX / g)
    cands = [base] + [base + s * k for k in range(1, search + 1) for s in (-1, 1)] \
        + [e4m3_bits(bmax / q / g) for q in (4.0, 3.0, 2.0)]
    best_err = best_s = best_c = None
    for bits in cands:
        cand = bits.clamp(8, 126).to(torch.uint8).view(torch.float8_e4m3fn)        # normals only
        cf = cand.to(torch.float32)
        v = (wb * ((1.0 / g) / cf).unsqueeze(-1)).clamp(-N.F4_MAX, N.F4_MAX)
        c = N.fp4_rne_code(v)
        err = ((N.fp4_code_to_f32(c) * (cf * g).unsqueeze(-1) - wb) ** 2).sum(-1)
        if best_err is None:
            best_err, best_s, best_c = err, cand, c
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s = torch.where(m, cand.view(torch.uint8), best_s.view(torch.uint8)).view(torch.float8_e4m3fn)
            best_c = torch.where(m.unsqueeze(-1), c, best_c)
    return (N.pack_fp4(best_c.view(R, C)).contiguous(),
            N.to_blocked_scales(best_s.view(torch.uint8)).contiguous(),
            g.reshape(()).to(torch.float32).contiguous())


def nvfp4_gemm(qa, sa, ga, qw, sw, gw):
    return F.scaled_mm(qa.view(_FP4X2), qw.view(_FP4X2).t(),
                       [sa.view(torch.float8_e4m3fn), ga], _REC,
                       [sw.view(torch.float8_e4m3fn), gw], _REC,
                       swizzle_a=_SWZ, swizzle_b=_SWZ, output_dtype=torch.bfloat16)


class NVFP4W4A4Linear(nn.Module):
    """Drop-in for a quantized bias-free nn.Linear: y = Q(x*col) @ Q(mat_q)^T.
    col=None for methods whose per-channel scale is folded offline (SmoothQuant)."""

    def __init__(self, mat_q, col=None, device="cuda"):
        super().__init__()
        n, k = mat_q.shape
        assert k % 64 == 0 and n % 16 == 0, (n, k)
        self.in_features, self.out_features = k, n
        qw, sw, gw = quantize_weight(mat_q.to(device))
        self.register_buffer("qweight", qw)        # [N, K/2] uint8 (FP4 x2)
        self.register_buffer("wscale", sw)         # [N*K/16] uint8 (E4M3, swizzled)
        self.register_buffer("wtscale", gw)        # [] fp32
        self.register_buffer("col", None if col is None else
                             col.to(device=device, dtype=torch.float32).contiguous())

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        if x2.dtype != torch.bfloat16 or not x2.is_contiguous():
            x2 = x2.to(torch.bfloat16).contiguous()
        qa, sa, ga = quantize_act(x2, self.col)
        y = nvfp4_gemm(qa, sa, ga, self.qweight, self.wscale, self.wtscale)
        return y.reshape(*shp[:-1], self.out_features)

    def dequant_weight(self):
        """For verification only: the exact weight the GEMM sees, as fp32 [N, K]."""
        n, k = self.out_features, self.in_features
        codes = torch.stack([self.qweight & 15, self.qweight >> 4], -1).reshape(n, k)
        nrb, ncb = triton.cdiv(n, 128), triton.cdiv(k // 16, 4)
        s = (self.wscale.view(nrb, ncb, 32, 4, 4).permute(0, 3, 2, 1, 4)
             .reshape(nrb * 128, ncb * 4)[:n, :k // 16].contiguous().view(torch.float8_e4m3fn))
        return N.nvfp4_dequant_ref(codes, s, self.wtscale)
