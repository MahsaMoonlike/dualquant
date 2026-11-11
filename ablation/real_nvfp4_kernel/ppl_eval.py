"""WikiText-2 PPL through the REAL packed NVFP4 W4A4 path (or BF16), same windowing as
fake_quant_ppl.py and ../real_int4_kernel/ppl_eval.py.

Usage:
    python ppl_eval.py --mode bf16  --seqlen 8192
    python ppl_eval.py --mode nvfp4 --seqlen 8192
"""
import argparse

import common
import torch

from full_model_nvfp4 import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--mode", choices=["bf16", "nvfp4"], required=True)
    ap.add_argument("--seqlen", type=int, default=8192)
    cli = ap.parse_args()
    import nvfp4_numerics as N
    dev = torch.device("cuda:0")
    model = build_model(cli.model, cli.mode, dev)
    model.config.use_cache = False
    ids = common.wikitext_test_ids(cli.model, cli.seqlen)
    print(f"{cli.mode}: test tokens={ids.numel()} windows={ids.numel()//cli.seqlen} seqlen={cli.seqlen}")
    val = common.ppl(model, ids, cli.seqlen, dev)
    tag = f"real packed W4A4, act NVFP4[{N.MODE}]" if cli.mode == "nvfp4" else "BF16"
    print(f"\nRESULT {cli.mode} WikiText-2 PPL (seqlen={cli.seqlen}, {tag}): {val:.4f}")


if __name__ == "__main__":
    main()
