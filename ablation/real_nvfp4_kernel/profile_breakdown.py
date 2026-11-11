"""Per-component GPU time (Table-11 style) for all four workloads, BF16 vs real NVFP4 W4A4.

Profiles the SAME CUDA-graph-captured forwards that graph_bench_all.py times. A graph replay
contains only raw kernels, so components are identified by kernel name:

  quantized linears : NVFP4 path -> cuBLASLt block-scaled FP4 GEMM kernels;
                      BF16 path  -> all BF16 cuBLAS GEMMs minus the lm_head share (below)
  act quantization  : dq_nvfp4_quantize_act_kernel (+ _act_amax_kernel in two-level mode);
                      beta is applied inside it
  lm_head (BF16)    : timed in its own captured graph on the identical input shape, because
                      in the BF16 model its GEMM is indistinguishable by name from the
                      projections'; the same value is used for both precisions
  attention         : SDPA kernels (flash / mem-efficient / cuDNN)
  other             : RMSNorm, RoPE, SiLU*mul, residual adds, KV-cache writes, casts, ...

Usage: python profile_breakdown.py --mode bf16|nvfp4 [--show-kernels]
"""
import argparse
import contextlib

import common
import torch
import transformers

import graph_bench_all as GBA
from full_model_nvfp4 import build_model

CATS = ["linears", "act_quant", "lm_head", "attention", "other"]


def kernel_cat(name):
    n = name.lower()
    if "quantize_act" in n or "act_amax" in n:
        return "act_quant"
    if "e2m1" in n or "nvf4" in n or "mxf4" in n or "fp4" in n or "blockscaled" in n or "bstensorop" in n:
        return "fp4_gemm"
    if "flash" in n or "fmha" in n or "attention" in n or "attn" in n or "cudnn" in n:
        return "attention"
    if "gemm" in n or "gemv" in n or "nvjet" in n or "cutlass" in n or "sm90_xmma" in n or "splitk" in n:
        return "bf16_gemm"
    return "other"


def profile_graph(g, reps, show=False):
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
    out = {}
    rows = []
    for e in prof.key_averages():
        t = e.self_device_time_total / reps / 1000.0      # ms per replay
        if t <= 0:
            continue
        c = kernel_cat(e.key)
        out[c] = out.get(c, 0.0) + t
        rows.append((t, c, e.key))
    if show:
        for t, c, k in sorted(rows, reverse=True)[:25]:
            print(f"      {t:8.3f} ms  {c:<10} {k[:110]}")
    return out


@torch.no_grad()
def lm_head_ms(model, shape, reps):
    h = torch.randn(*shape, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    f = lambda: model.lm_head(h)
    GBA._warm_on_side_stream(f)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    return sum(profile_graph(g, reps).values())


def split(d, lm):
    """kernel categories -> table rows."""
    gemm = d.get("fp4_gemm", 0.0) + d.get("bf16_gemm", 0.0)
    return {"linears": gemm - lm, "act_quant": d.get("act_quant", 0.0), "lm_head": lm,
            "attention": d.get("attention", 0.0), "other": d.get("other", 0.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--mode", choices=["bf16", "nvfp4"], required=True)
    ap.add_argument("--show-kernels", action="store_true")
    cli = ap.parse_args()
    dev = torch.device("cuda:0")
    model = build_model(cli.model, cli.mode, dev)
    tok = transformers.AutoTokenizer.from_pretrained(cli.model)
    ids = tok(common.PROMPT, return_tensors="pt").input_ids.to(dev)
    P = ids.shape[1]

    print(f"\n===== {cli.mode.upper()} per-component GPU time (ms per forward) =====")
    print(f"{'workload':<14}" + "".join(f"{c:>12}" for c in CATS) + f"{'TOTAL':>10}")
    results = {}
    with torch.no_grad():
        for bs in (1, 16):
            g, _, _ = GBA.capture_prefill(model, ids, bs)
            d = profile_graph(g, 20, cli.show_kernels)
            row = split(d, lm_head_ms(model, (bs, P), 20))
            results[f"prefill b{bs}"] = row
            del g
            print(f"{'prefill b'+str(bs):<14}" + "".join(f"{row[c]:>12.3f}" for c in CATS)
                  + f"{sum(row.values()):>10.3f}", flush=True)
        for bs in (1, 16):
            st = GBA.DecodeStep(model, ids, bs, P + 180, dev)
            GBA._warm_on_side_stream(st)
            st.reset(P + 100)          # profile one step at the 182-token cache
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                st()
            d = profile_graph(g, 50, cli.show_kernels)
            row = split(d, lm_head_ms(model, (bs, 1), 50))
            results[f"decode b{bs}"] = row
            del g, st
            print(f"{'decode b'+str(bs):<14}" + "".join(f"{row[c]:>12.3f}" for c in CATS)
                  + f"{sum(row.values()):>10.3f}", flush=True)
    if cli.mode == "nvfp4":
        print("\nactivation quantization share of NVFP4 GPU time: " + ", ".join(
            f"{k} {100*v['act_quant']/sum(v.values()):.1f}%" for k, v in results.items()))


if __name__ == "__main__":
    main()
