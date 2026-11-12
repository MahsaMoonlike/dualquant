"""Convert Llama-3.1-8B to the real packed NVFP4 W4A4 DualQuant path.

Every one of the 224 projections (q/k/v/o/gate/up/down x 32 blocks) becomes an
NVFP4W4A4Linear (nvfp4_kernels.py): packed FP4 weights + swizzled E4M3 block scales +
FP32 tensor scale, activations quantized at runtime by the fused beta+NVFP4 Triton kernel,
cuBLASLt block-scaled FP4 GEMM. Embedding, lm_head and norms stay BF16.

DualQuant's (mat_q, column scales) come from dq_nvfp4_bcd (cached in wrap_cache/), i.e. the
same tensors fake_quant_ppl.py uses, so fake vs real differ only by the packing.
"""
import time

import common
import torch

from dq_nvfp4_bcd import dualquant_nvfp4_cached
from nvfp4_kernels import NVFP4W4A4Linear


@torch.no_grad()
def convert_model_to_nvfp4(model, model_id, dev="cuda", verbose=True):
    t0 = time.time()
    for i, parent, attr, key in common.iter_linears(model):
        lin = getattr(parent, attr)
        lin.to(dev)
        col = dualquant_nvfp4_cached(lin, model_id, key)        # lin.weight <- mat_q (bf16)
        # pack on the GPU, then park on the CPU: the final model.to(dev) then allocates all
        # packed buffers in one go, instead of interleaving them with the conversion's large
        # temporaries (which fragments the caching allocator by ~1.1 GB that empty_cache
        # cannot return)
        setattr(parent, attr, NVFP4W4A4Linear(lin.weight.data, col, device=dev).cpu())
        del lin
        if verbose and attr == "down_proj" and i % 8 == 7:
            print(f"  converted through layer {i}  ({time.time()-t0:.0f}s)", flush=True)
    torch.cuda.empty_cache()
    return model


def smoothquant_act_scales_path(model_id):
    """The paper's calibrated SmoothQuant activation statistics shipped in the repo
    (smoothquant/act_scales/, WikiText-2 train, seq 8192). Override with DQ_SQ_SCALES."""
    import glob
    import os
    if os.environ.get("DQ_SQ_SCALES"):
        return os.environ["DQ_SQ_SCALES"]
    name = model_id.split("/")[-1].replace("Meta-", "")
    hits = glob.glob(os.path.join(common.ROOT, "smoothquant", "act_scales", f"{name}_seq*_4bit.pt"))
    if len(hits) != 1:
        raise FileNotFoundError(f"no unique SmoothQuant act scales for {name}: {hits}")
    return hits[0]


def apply_smoothquant(model, model_id, alpha=0.5):
    """Upstream SmoothQuant recipe for Llama/Qwen-style blocks (repo's smooth_lm, same as
    mit-han-lab/smoothquant smooth.py): one shared scale for q/k/v folded into input_layernorm,
    one for gate/up folded into post_attention_layernorm; o_proj/down_proj unsmoothed.
    alpha=0.5 as in the paper's baseline. Zero runtime cost: nothing left to apply online."""
    from smoothquant.smooth_with_scale_dict import smooth_lm
    smooth_lm(model, torch.load(smoothquant_act_scales_path(model_id)), alpha)
    return model


@torch.no_grad()
def apply_smoothquant_sweep(model, model_id, ncalib=None):
    """SmoothQuant with per-smoothing-point alpha from calib_cost.py --method sq (act absmax
    over the calibration set, alpha in {0.4, 0.5, 0.6} chosen per layer), folded into the
    RMSNorms by the same smooth_ln_fcs_llama_like the repo's smooth_lm uses."""
    import os
    from calib_cost import sq_sweep_path
    from smoothquant.smooth_with_scale_dict import smooth_ln_fcs_llama_like
    ncalib = ncalib or int(os.environ.get("DQ_SQ_NCALIB", "128"))
    d = torch.load(sq_sweep_path(model_id, ncalib))
    for i, blk in enumerate(model.model.layers):
        a = blk.self_attn
        smooth_ln_fcs_llama_like(blk.input_layernorm, [a.q_proj, a.k_proj, a.v_proj],
                                 d["act_amax"][f"layer{i}.attn"], d["alpha"][f"layer{i}.attn"])
        smooth_ln_fcs_llama_like(blk.post_attention_layernorm, [blk.mlp.gate_proj, blk.mlp.up_proj],
                                 d["act_amax"][f"layer{i}.mlp"], d["alpha"][f"layer{i}.mlp"])
    return model


@torch.no_grad()
def convert_model_to_sq_nvfp4(model, model_id, dev="cuda", verbose=True, sweep=False):
    """SmoothQuant + NVFP4 RTN weights (the paper's SmoothQuant baseline), packed into the same
    NVFP4W4A4Linear with col=None, i.e. activation quantization without a per-channel multiply.
    sweep=True: calib_cost.py's per-layer alpha sweep instead of the repo's scales at 0.5."""
    import nvfp4_numerics as N
    (apply_smoothquant_sweep if sweep else apply_smoothquant)(model, model_id)
    for i, parent, attr, key in common.iter_linears(model):
        lin = getattr(parent, attr)
        w = lin.weight.data.to(dev)
        mat_q = N.nvfp4_fake_quant(w.float()).to(w.dtype)       # RTN NVFP4 of smoothed W
        setattr(parent, attr, NVFP4W4A4Linear(mat_q, None, device=dev).cpu())
        del lin, w
        if verbose and attr == "down_proj" and i % 8 == 7:
            print(f"  converted through layer {i}", flush=True)
    torch.cuda.empty_cache()
    return model


@torch.no_grad()
def convert_model_to_awq_nvfp4(model, model_id, dev="cuda", verbose=True):
    """AWQ+ (awq_nvfp4.py): packed Q(W s), activations scaled by 1/s inside the fused
    activation-quant kernel (same mechanism as DualQuant's beta). Scales must be cached
    first (python awq_nvfp4.py), so that no calibration runs inside a measured build."""
    import awq_nvfp4 as A
    if not A.all_cached(model_id, model.config):
        raise RuntimeError("AWQ scales not cached: run `python awq_nvfp4.py` first")
    for i, parent, attr, key in common.iter_linears(model):
        mat_q, col, _ = A.load_awq(model_id, key)
        setattr(parent, attr, NVFP4W4A4Linear(mat_q.to(dev), col, device=dev).cpu())
        if verbose and attr == "down_proj" and i % 8 == 7:
            print(f"  converted through layer {i}", flush=True)
    torch.cuda.empty_cache()
    return model


def build_model(model_id, mode, dev="cuda"):
    """mode: 'bf16', 'nvfp4' (DualQuant), 'sq_nvfp4' (SmoothQuant), 'sqs_nvfp4' (SmoothQuant,
    per-layer alpha sweep) or 'awq_nvfp4' (AWQ+).
    Eval-mode model on `dev`."""
    model = common.load_bf16(model_id)
    if mode == "nvfp4":
        convert_model_to_nvfp4(model, model_id, dev)
    elif mode == "sq_nvfp4":
        convert_model_to_sq_nvfp4(model, model_id, dev)
    elif mode == "sqs_nvfp4":
        convert_model_to_sq_nvfp4(model, model_id, dev, sweep=True)
    elif mode == "awq_nvfp4":
        convert_model_to_awq_nvfp4(model, model_id, dev)
    return model.to(dev).eval()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    cli = ap.parse_args()
    m = build_model(cli.model, "nvfp4")
    n = sum(isinstance(x, NVFP4W4A4Linear) for x in m.modules())
    print(f"converted {n} projections to NVFP4 W4A4")
