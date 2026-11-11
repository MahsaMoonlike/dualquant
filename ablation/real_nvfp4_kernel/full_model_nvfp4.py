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


def build_model(model_id, mode, dev="cuda"):
    """mode: 'bf16' or 'nvfp4'. Returns an eval-mode model on `dev`."""
    model = common.load_bf16(model_id)
    if mode == "nvfp4":
        convert_model_to_nvfp4(model, model_id, dev)
    return model.to(dev).eval()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    cli = ap.parse_args()
    m = build_model(cli.model, "nvfp4")
    n = sum(isinstance(x, NVFP4W4A4Linear) for x in m.modules())
    print(f"converted {n} projections to NVFP4 W4A4")
