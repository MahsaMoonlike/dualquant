"""Print the measured logs as the paper's tables (markdown).

  python tables.py memory  [model_dir] [--quant nvfp4|sq_nvfp4|awq_nvfp4]   # Table 12 layout
Reads logs/<model_dir>/memory_bf16.log and memory_<quant>.log written by memory_breakdown.py.
"""
import argparse
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
WORKLOADS = ["prefill b=1", "prefill b=16", "decode b=1", "decode b=16"]
HEADS = ["Prefill (b=1)", "Prefill (b=16)", "Decode (b=1)", "Decode (b=16)"]


def parse_memory(path):
    txt = open(path).read()
    txt = txt[txt.index("memory breakdown (GB)"):]
    comp = {}
    for label, key in [("Quantized-linear weights", "lin_w"), ("Per-16-group weight scales", "scales"),
                       ("Per-tensor scales", "misc"), ("Embedding + lm_head", "embed"),
                       ("Norms", "norms"), ("Runtime/allocator overhead", "overhead"),
                       ("== Weights subtotal", "weights"), ("First-forward scratch", "scratch")]:
        m = re.search(re.escape(label) + r".*?\s(-?\d+\.\d+)\s*$", txt, re.M)
        comp[key] = float(m.group(1)) if m else None
    rows = {}
    for w in WORKLOADS:
        m = re.search(re.escape(w) + r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.-]+)\s+([\d.]+)", txt)
        rows[w] = dict(zip(["weights", "scratch", "kv", "other", "total"], map(float, m.groups())))
    return comp, rows


def memory_table(model_dir, quant):
    d = os.path.join(_HERE, "logs", model_dir)
    b, br = parse_memory(os.path.join(d, "memory_bf16.log"))
    q, qr = parse_memory(os.path.join(d, f"memory_{quant}.log"))
    f = lambda v: "–" if v is None else (f"{v:.2f}" if abs(v) >= 0.005 else "0")
    # norms are tiny (~1 MB); folded into the overhead row, as in the paper's table
    ovh = lambda c: (c["overhead"] or 0) + (c["norms"] or 0)
    rows = [
        ("Quantized linear weights", b["lin_w"], q["lin_w"]),
        ("Per-group weight scales (FP8 E4M3)", None, q["scales"]),
        ("Per-tensor scales + β (FP32)", None, q["misc"]),
        ("Embedding + lm_head (BF16)", b["embed"], q["embed"]),
        ("Runtime/allocator overhead", ovh(b), ovh(q)),
        ("**Weights subtotal**", b["weights"], q["weights"]),
        ("First-forward scratch", b["scratch"], q["scratch"]),
    ]
    out = ["| Component | " + " | ".join(f"{h} BF16 | {h} W4A4" for h in HEADS) + " |",
           "|---|" + "---|" * 8]
    for name, bv, qv in rows:
        out.append(f"| {name} | " + " | ".join(f"{f(bv)} | {f(qv)}" for _ in WORKLOADS) + " |")
    for name, key in [("KV cache", "kv"), ("Other activations", "other"), ("**Total**", "total")]:
        out.append(f"| {name} | " + " | ".join(f"{f(br[w][key])} | {f(qr[w][key])}" for w in WORKLOADS) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("table", choices=["memory"])
    ap.add_argument("model_dir", nargs="?", default="Meta-Llama-3.1-8B")
    ap.add_argument("--quant", default="nvfp4")
    a = ap.parse_args()
    print(memory_table(a.model_dir, a.quant))
