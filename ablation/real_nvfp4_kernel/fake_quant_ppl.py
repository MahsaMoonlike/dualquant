"""WikiText-2 PPL with FAKE NVFP4 quantization, built from the repo's own pieces
(formats.make_format("nvfp4") for activations, activations.install_perm_linear for the
runtime beta pre-scale + activation cast), so it is the Table-1 style number that the real
packed path in ppl_eval.py must reproduce.

  --method bf16       no quantization
  --method rtn        W: NVFP4 RTN, A: NVFP4 RTN
  --method dualquant  W: DualQuant (mat_q), A: NVFP4 of (x * column_scales)
  --weight-only       keep activations in BF16 (Table-10 style)

Usage: python fake_quant_ppl.py --method dualquant --seqlen 8192
"""
import argparse
import time

import common
import torch

from activations import install_perm_linear
from formats import make_format


def build(model_id, method, weight_only, dev):
    model = common.load_bf16(model_id)
    if method == "bf16":
        return model.to(dev)
    fmt = make_format("nvfp4")
    t0 = time.time()
    with torch.no_grad():
        for i, parent, attr, key in common.iter_linears(model):
            lin = getattr(parent, attr)
            if method == "rtn":
                lin.weight.data.copy_(fmt.cast(lin.weight.data.to(dev).float()).to(lin.weight.dtype).cpu())
                col = None
            else:
                from dq_nvfp4_bcd import dualquant_nvfp4_cached
                col = dualquant_nvfp4_cached(lin, model_id, key)
                if weight_only:   # fold the column scales back into the weight
                    lin.weight.data.copy_((lin.weight.data.float() * col[None]).to(lin.weight.dtype))
            if not weight_only:
                install_perm_linear(parent, attr, act_fmt=fmt, col_scales=col)
            if attr == "down_proj" and (i % 8 == 7):
                print(f"  quantized through layer {i}  ({time.time()-t0:.0f}s)", flush=True)
    return model.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--method", choices=["bf16", "rtn", "dualquant"], required=True)
    ap.add_argument("--weight-only", action="store_true")
    ap.add_argument("--seqlen", type=int, default=8192)
    cli = ap.parse_args()
    dev = torch.device("cuda:0")
    import nvfp4_numerics as N
    model = build(cli.model, cli.method, cli.weight_only, dev).eval()
    model.config.use_cache = False
    ids = common.wikitext_test_ids(cli.model, cli.seqlen)
    val = common.ppl(model, ids, cli.seqlen, dev)
    tag = "W-only" if cli.weight_only else "W4A4"
    print(f"\nRESULT fake-quant {cli.method} {tag if cli.method!='bf16' else ''} NVFP4[{N.MODE}] "
          f"WikiText-2 PPL (seqlen={cli.seqlen}): {val:.4f}")


if __name__ == "__main__":
    main()
