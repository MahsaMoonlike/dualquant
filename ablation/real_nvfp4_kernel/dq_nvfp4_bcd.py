"""DualQuant BCD for NVFP4, batched over all 16-column blocks on the GPU, plus a disk cache.

This is the same algorithm as layer_wrapper_dualquant.wrap_dualquant_layer with the repo's
configs/methods.json "dualquant" settings (scale_option=row_column, row_init=max_abs,
col_init=l1_norm, num_iter=15, num_splits=1, dualquant_to_element_tensor=false) and
--weight-scale-format e4m3, which is what the NVFP4 rows of the paper use.

The reference loops over the n_cols/16 blocks one at a time on the CPU. The blocks are
independent, so here they are stacked into one [n_blocks, n_rows, 16] tensor and every line
of dualquant_with_alpha_and_beta / _reconstruct_block_alpha_beta is applied to all of them
at once. `inner cast` = the block-format cast (Format "NVFP4[E2M1]{16}"), which for a
[rows,16] block_matrix takes its per-tensor amax over that block_matrix; batched, that is the
amax over dims (1,2) of each batch entry (batch_dims=1).

verification/test_bcd_matches_reference.py checks this against the reference loop.
The cache stores (mat_q, column_scales) exactly like ../real_int4_kernel/wrap_cache.py.
"""
import hashlib
import os

import torch

import nvfp4_numerics as N

EPS = 1e-12
# cuda (default, ~1 min for Llama-3.1-8B) or cpu (bit-identical to the repo's per-block loop,
# much slower). Both give the same reconstruction MSE; GPU differs only in summation order.
BCD_DEVICE = os.environ.get("DQ_BCD_DEVICE", "cuda")
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrap_cache")


def _q_e4m3(s):          # scale_formats.quantize_scale_e4m3
    return s.clamp(min=N.E4M3_TINY, max=N.E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def _cast(x):            # inner/final block quantiser, one independent tensor per batch entry
    return N.nvfp4_fake_quant(x, batch_dims=1)


@torch.no_grad()
def dualquant_nvfp4(weight: torch.Tensor, num_iter=15, device=None):
    """weight [n_rows, n_cols] -> (mat_q [n_rows, n_cols] fp32, column_scales [n_cols] fp32)."""
    W2 = weight.detach().to(device=device or BCD_DEVICE, dtype=torch.float32)
    R, C = W2.shape
    assert C % 16 == 0
    nb = C // 16
    W = W2.reshape(R, nb, 16).permute(1, 0, 2).contiguous()        # [nb, R, 16]

    # init (_init_beta l1_norm, _init_alpha max_abs with max_val=6)
    beta = 1.0 / W.abs().sum(dim=1).clamp(min=1e-8)                 # [nb, 16]
    bn = W * beta[:, None, :]
    alpha = 1.0 / (bn.abs().amax(-1) / torch.tensor(6.0))           # [nb, R]
    alpha_inv = 1.0 / alpha
    beta_inv = 1.0 / beta

    for _ in range(num_iter):
        M = _cast(alpha[:, :, None] * W * beta[:, None, :])
        denom = ((M * beta_inv[:, None, :]) ** 2).sum(-1) + EPS
        alpha_inv = (W * M * beta_inv[:, None, :]).sum(-1) / denom
        alpha_inv = _q_e4m3(alpha_inv)
        alpha = 1.0 / alpha_inv

        M = _cast(alpha[:, :, None] * W * beta[:, None, :])
        denom = ((M * alpha_inv[:, :, None]) ** 2).sum(1) + EPS
        beta_inv = (W * M * alpha_inv[:, :, None]).sum(1) / denom
        beta = 1.0 / beta_inv

    Wq = _cast(alpha[:, :, None] * W * beta[:, None, :])
    alpha_inv_q = _q_e4m3(1.0 / alpha)
    blocks = alpha_inv_q[:, :, None] * Wq                           # [nb, R, 16]
    mat_q = blocks.permute(1, 0, 2).reshape(R, C)
    col = (1.0 / beta).reshape(C)
    if not (torch.isfinite(mat_q).all() and torch.isfinite(col).all()):
        raise FloatingPointError("non-finite DualQuant output")
    return mat_q, col


GAUGE = os.environ.get("DQ_BETA_GAUGE", "none")


def apply_gauge(mat_q, col):
    """DualQuant's (alpha, beta) has a free overall factor (paper, footnote 2): W is unchanged
    by col -> col / c, mat_q -> mat_q * c. With c a power of two this is EXACT for NVFP4: every
    FP4 code and E4M3 scale is unchanged, both on the weight side (packed with a power-of-two
    tensor scale) and on the activation side -- except that activation block scales move by
    1/c, i.e. out of E4M3's saturation at 448. Calibration-free: c depends on col only.
      none   : keep the BCD's own normalisation (col ~ column l1 norm, up to ~750 on Qwen3)
      median : c = 2^round(log2(median(col)))  -> typical channels are left unscaled"""
    if GAUGE == "none":
        return mat_q, col
    assert GAUGE == "median", GAUGE
    k = torch.round(torch.log2(col.float().median()))
    c = torch.exp2(k).to(col.dtype)
    return mat_q * c.to(mat_q.dtype), col / c


def _cache_path(model_id, layer_key):
    safe = model_id.replace("/", "__")
    tag = f"{model_id}|{layer_key}|nvfp4|bs16|{N.MODE}|{BCD_DEVICE}"
    h = hashlib.sha1(tag.encode()).hexdigest()[:8]
    return os.path.join(_CACHE_DIR, f"{safe}__{layer_key}__nvfp4_{N.MODE}_{BCD_DEVICE}__{h}.pt")


def dualquant_nvfp4_cached(module, model_id, layer_key, num_iter=15):
    """Same contract as wrap_cache.wrap_cached: overwrites module.weight with mat_q (in the
    module's dtype, as the repo's wrapper does) and returns the column scales."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _cache_path(model_id, layer_key)
    if os.path.exists(path):
        d = torch.load(path, map_location="cpu")
        mat_q, col = d["mat_q"], d["beta"]
    else:
        mat_q, col = dualquant_nvfp4(module.weight, num_iter=num_iter)
        mat_q, col = mat_q.cpu(), col.cpu()
        torch.save({"mat_q": mat_q, "beta": col}, path)
    mat_q, col = apply_gauge(mat_q, col)
    module.weight.data.copy_(mat_q.to(module.weight.dtype).to(module.weight.device))
    return col.to(module.weight.device)
