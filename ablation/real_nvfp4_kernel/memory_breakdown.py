"""End-to-end GPU memory breakdown (Table-12 style), BF16 vs real packed NVFP4 W4A4.

Unlike the INT4/nunchaku path, every buffer here (packed weights, scales, activation
operands, cuBLAS workspace) is a torch allocation, so torch's counters see everything.
mem_get_info is still used for the resident total, so the allocator's reserve and the
CUDA/cuBLAS/Triton runtime are counted rather than hidden.

  weights subtotal   = mem_get_info delta after load + empty_cache (everything resident)
    = packed linear weights + per-16 E4M3 scales + per-tensor scales & beta + embedding
      + lm_head + norms + runtime/allocator overhead (the measured remainder)
  first-forward scratch = persistent growth after the first forward (cuBLASLt workspace,
                          kernel modules), measured after empty_cache
  KV cache           = analytic, BF16, (prompt + 100) tokens
  other activations  = torch peak during the workload - resident torch bytes - KV cache
  total              = weights + scratch + transient peak

Workloads: prefill = one forward over the 82-token prompt without KV cache; decode =
generate(max_new_tokens=100) (182-token cache), batch 1 and 16, as in the paper.

Usage: python memory_breakdown.py --mode bf16|nvfp4
"""
import argparse

import common
import torch
import transformers

from full_model_nvfp4 import build_model

GB = common.GB


def tensor_bytes(model):
    from nvfp4_kernels import NVFP4W4A4Linear
    c = dict(lin_w=0, lin_scale=0, lin_misc=0, embed=0, lm_head=0, norms=0)
    q_ids = set()
    for m in model.modules():
        if isinstance(m, NVFP4W4A4Linear):
            c["lin_w"] += m.qweight.numel()
            c["lin_scale"] += m.wscale.numel()
            c["lin_misc"] += m.wtscale.numel() * 4 + (0 if m.col is None else m.col.numel() * 4)
            q_ids.update(id(b) for b in m.buffers())
    for i, parent, attr, _ in common.iter_linears(model):
        lin = getattr(parent, attr)
        if isinstance(lin, torch.nn.Linear):
            c["lin_w"] += lin.weight.numel() * lin.weight.element_size()
            q_ids.add(id(lin.weight))
    c["embed"] = model.model.embed_tokens.weight.numel() * 2
    c["lm_head"] = model.lm_head.weight.numel() * 2
    seen = {id(model.model.embed_tokens.weight), id(model.lm_head.weight)} | q_ids
    for p in list(model.parameters()) + list(model.buffers()):
        if id(p) not in seen:
            c["norms"] += p.numel() * p.element_size()
            seen.add(id(p))
    return {k: v / GB for k, v in c.items()}


def resident(total, base):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    f, _ = torch.cuda.mem_get_info()
    return (total - f) / GB - base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--mode", choices=["bf16", "nvfp4", "sq_nvfp4", "awq_nvfp4"], required=True)
    cli = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.cuda.init()
    torch.zeros(1, device=dev)
    f0, total = torch.cuda.mem_get_info()
    base = (total - f0) / GB                       # CUDA context, excluded

    model = build_model(cli.model, cli.mode, dev)
    tok = transformers.AutoTokenizer.from_pretrained(cli.model)
    weights = resident(total, base)
    torch_resident = torch.cuda.memory_allocated() / GB
    tb = tensor_bytes(model)
    overhead = weights - sum(tb.values())

    ids = tok(common.PROMPT, return_tensors="pt").input_ids.to(dev)
    P, NEW = ids.shape[1], 100
    c = model.config
    hd = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    kv = lambda bs: bs * (P + NEW) * 2 * c.num_key_value_heads * hd * 2 * c.num_hidden_layers / GB

    rows = []
    with torch.no_grad():
        model(ids, use_cache=False)                       # first forward: workspaces etc.
        scratch = resident(total, base) - weights
        for bs in (1, 16):
            ids_b = ids.repeat(bs, 1)
            model(ids_b, use_cache=False)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            model(ids_b, use_cache=False)
            torch.cuda.synchronize()
            tr = torch.cuda.max_memory_allocated() / GB - torch_resident
            rows.append((f"prefill b={bs}", 0.0, tr))
        for bs in (1, 16):
            ids_b = ids.repeat(bs, 1)
            gen = dict(attention_mask=torch.ones_like(ids_b), max_new_tokens=NEW, min_new_tokens=NEW,
                       do_sample=False, pad_token_id=tok.eos_token_id)
            model.generate(ids_b, **gen)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            model.generate(ids_b, **gen)
            torch.cuda.synchronize()
            tr = torch.cuda.max_memory_allocated() / GB - torch_resident
            rows.append((f"decode b={bs}", kv(bs), tr))
        scratch = max(scratch, resident(total, base) - weights)

    q = cli.mode != "bf16"
    print(f"\n===== {cli.mode.upper()} memory breakdown (GB) — {cli.model} =====")
    print(f"  {'Quantized-linear weights' + (' (FP4 packed)' if q else ' (BF16)'):<44}{tb['lin_w']:8.3f}")
    if q:
        print(f"  {'Per-16-group weight scales (FP8 E4M3)':<44}{tb['lin_scale']:8.3f}")
        print(f"  {'Per-tensor scales + beta (FP32)':<44}{tb['lin_misc']:8.3f}")
    print(f"  {'Embedding + lm_head (BF16)':<44}{tb['embed'] + tb['lm_head']:8.3f}")
    print(f"  {'Norms (BF16)':<44}{tb['norms']:8.3f}")
    print(f"  {'Runtime/allocator overhead (measured rest)':<44}{overhead:8.3f}")
    print(f"  {'== Weights subtotal (mem_get_info)':<44}{weights:8.3f}")
    print(f"  {'First-forward scratch':<44}{scratch:8.3f}")
    print(f"\n  {'workload':<14}{'weights':>9}{'scratch':>9}{'KVcache':>9}{'otheract':>9}{'TOTAL':>9}")
    for name, kvb, tr in rows:
        print(f"  {name:<14}{weights:9.3f}{scratch:9.3f}{kvb:9.3f}{tr - kvb:9.3f}"
              f"{weights + scratch + tr:9.3f}")


if __name__ == "__main__":
    main()
