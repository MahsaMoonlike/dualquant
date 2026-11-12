"""Offline cost of each method's scale-finding phase (NVFP4 W4A4), one method per process.

  sq   SmoothQuant calibration: per-channel act absmax of the q/k/v and gate/up inputs over the
       calibration set, plus a per-layer sweep of the migration strength over the SmoothQuant
       paper's sweet-spot region alpha in {0.4, 0.5, 0.6} (Sec. 5.5). Each smoothing point
       (input_layernorm -> q/k/v, post_attention_layernorm -> gate/up) keeps the alpha that
       minimises sum_fc ||X W^T - Q(X/s) Q(W s)^T||^2 on the calibration tokens, with NVFP4
       quantization of both operands. o_proj/down_proj unsmoothed, as in SmoothQuant.
  awq  AWQ+ calibration, as the repo runs it (methods/awq.py -> legacy compute_awq_scale):
       per linear, 20-point grid over a in s = mu_x^a / mu_w^(1-a), minimising
       ||X W^T - X (Q(W s)/s)^T||, no clipping search. Then packs Q(W s).
  dq   DualQuant: the BCD over (alpha, beta) for every linear, T = 15 iterations (paper,
       Algorithm 1 / Appendix C). No data, no forward passes.

Calibration (sq, awq): the repo's AWQ calibration chunks (WikiText-2 train, 512 tokens),
exactly --ncalib of them (default 128). The calibration activations are those of the BF16
model, collected block by block (only one decoder block on the GPU at a time); both
calibrated methods use the same propagation code, so they differ only in their search.

Timed: everything after the BF16 model is in host memory (model load from disk is reported
separately and excluded for all methods), with CUDA synchronisation at phase boundaries.
Outputs are cached so the real packed path can evaluate them:
  sq  -> wrap_cache/<model>__sq_sweep_n<N>.pt   (build_model mode 'sqs_nvfp4')
  awq -> awq_nvfp4 cache, _n<N> variant         (mode 'awq_nvfp4' with DQ_AWQ_NCALIB=<N>)
  dq  -> compared against the dq_nvfp4_bcd cache (mode 'nvfp4'), must be identical

Usage: python calib_cost.py --method sq|awq|dq [--model ...] [--ncalib 128]
"""
import argparse
import os
import time

import common
import torch

import nvfp4_numerics as N

SQ_ALPHAS = (0.4, 0.5, 0.6)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrap_cache")


def sq_sweep_path(model_id, ncalib):
    return os.path.join(_CACHE_DIR, f"{model_id.replace('/', '__')}__sq_sweep_{N.MODE}_n{ncalib}.pt")


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        tm = self

        class _C:
            def __enter__(self):
                torch.cuda.synchronize()
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                torch.cuda.synchronize()
                tm.t[name] = tm.t.get(name, 0.0) + time.perf_counter() - self.t0
        return _C()


def sq_scales(act_amax, fcs, alpha):
    """smooth_ln_fcs_llama_like's scale, computed in the model dtype exactly as it does."""
    dtype = fcs[0].weight.dtype
    a = act_amax.to(dtype)
    w = torch.cat([fc.weight.abs().max(dim=0)[0].clamp(min=1e-5)[None] for fc in fcs]).max(0)[0]
    return (a.pow(alpha) / w.pow(1 - alpha)).clamp(min=1e-5).to(dtype)


@torch.no_grad()
def sq_errors(X, fcs, scales, rows=4096):
    """{alpha: sum_fc ||X W^T - Q(X/s) Q(W s)^T||^2}, NVFP4 fake quant on both operands.
    One pass over the tokens: each chunk's reference output X W^T is shared by all alphas."""
    Wf = [fc.weight.float() for fc in fcs]
    Wq = {a: [N.nvfp4_fake_quant((fc.weight * s[None, :]).float()) for fc in fcs]
          for a, s in scales.items()}
    err = {a: 0.0 for a in scales}
    for r in range(0, X.shape[0], rows):
        x = X[r:r + rows].float()
        xq = {a: N.nvfp4_fake_quant(x / s.float()) for a, s in scales.items()}
        for k, wf in enumerate(Wf):
            ref = x @ wf.T
            for a in scales:
                err[a] += (ref - xq[a] @ Wq[a][k].T).square().sum().item()
    return err


@torch.no_grad()
def run_calibrated(method, model, model_id, ncalib, tm, dev="cuda"):
    import transformers
    import awq_nvfp4 as A
    from awq import compute_awq_scale

    with tm("calib data"):
        tok = transformers.AutoTokenizer.from_pretrained(model_id)
        calib = A.calib_set(tok, ncalib)
    print(f"calibration: {len(calib)} x {calib[0].shape[1]} tokens", flush=True)

    with tm("forward (block-sequential)"):
        model.model.embed_tokens.to(dev)
        model.model.rotary_emb.to(dev)
        hs = [model.model.embed_tokens(x.to(dev)) for x in calib]
        pos = torch.arange(hs[0].shape[1], device=dev)[None]
        pe = model.model.rotary_emb(hs[0], pos)

    if method == "awq":
        names = [("self_attn", n) for n in common.ATTN] + [("mlp", n) for n in common.MLP]
    else:   # the two smoothing points' inputs (q/k/v share one, gate/up share one)
        names = [("self_attn", "q_proj"), ("mlp", "gate_proj")]
    out = {"act_amax": {}, "alpha": {}, "err": {}}
    fq = lambda w: N.nvfp4_fake_quant(w.to(dev))

    for i, blk in enumerate(model.model.layers):
        with tm("forward (block-sequential)"):
            blk.to(dev)
            acts = {n: [] for _, n in names}
            hooks = [getattr(getattr(blk, p), n).register_forward_hook(
                lambda m, inp, o, n=n: acts[n].append(inp[0].detach())) for p, n in names]
            for j in range(len(hs)):
                o = blk(hs[j], attention_mask=None, position_ids=pos, position_embeddings=pe)
                hs[j] = o[0] if isinstance(o, tuple) else o
            for h in hooks:
                h.remove()

        with tm("search"):
            if method == "awq":
                for p, n in names:
                    lin = getattr(getattr(blk, p), n)
                    W = lin.weight.data.float()
                    X = torch.cat(acts[n], 0)
                    acts[n] = None
                    s = compute_awq_scale(W, X, fq).float()
                    mat_q = N.nvfp4_fake_quant(W * s[None, :]).to(torch.bfloat16)
                    torch.save({"mat_q": mat_q.cpu(), "col": (1.0 / s).cpu(), "s": s.cpu()},
                               A._cache_path(model_id, f"layer{i}.{p}.{n}", ncalib))
                    del X, W
            else:
                groups = [("attn", "q_proj", [blk.self_attn.q_proj, blk.self_attn.k_proj,
                                              blk.self_attn.v_proj]),
                          ("mlp", "gate_proj", [blk.mlp.gate_proj, blk.mlp.up_proj])]
                for g, n, fcs in groups:
                    X = torch.cat(acts[n], 0).reshape(-1, fcs[0].in_features)
                    acts[n] = None
                    amax = X.abs().amax(0).float()                    # SmoothQuant act scale
                    errs = sq_errors(X, fcs, {a: sq_scales(amax, fcs, a) for a in SQ_ALPHAS})
                    best = min(errs, key=errs.get)
                    key = f"layer{i}.{g}"
                    out["act_amax"][key] = amax.cpu()
                    out["alpha"][key] = best
                    out["err"][key] = errs
                    del X
        with tm("forward (block-sequential)"):
            blk.cpu()
            del acts
            torch.cuda.empty_cache()
        if i % 8 == 7:
            extra = "" if method == "awq" else \
                f"  alphas {[out['alpha'][f'layer{k}.{g}'] for k in range(i-7, i+1) for g in ('attn', 'mlp')]}"
            print(f"  layer {i} done ({sum(tm.t.values()):.0f}s){extra}", flush=True)

    if method == "sq":
        torch.save(out, sq_sweep_path(model_id, ncalib))
        from collections import Counter
        print("chosen alphas:", dict(Counter(out["alpha"].values())))


@torch.no_grad()
def run_dualquant(model, model_id, tm):
    from dq_nvfp4_bcd import _cache_path, dualquant_nvfp4
    checked = mism = 0
    for i, parent, attr, key in common.iter_linears(model):
        with tm("BCD (T=15)"):
            mat_q, col = dualquant_nvfp4(getattr(parent, attr).weight, num_iter=15)
            mat_q, col = mat_q.cpu(), col.cpu()
        p = _cache_path(model_id, key)
        if os.path.exists(p):                         # same tensors the real path deploys
            d = torch.load(p, map_location="cpu")
            checked += 1
            mism += not (torch.equal(d["mat_q"], mat_q) and torch.equal(d["beta"], col))
        if attr == "down_proj" and i % 8 == 7:
            print(f"  layer {i} done ({sum(tm.t.values()):.0f}s)", flush=True)
    print(f"cache check: {checked} linears compared, {mism} differ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sq", "awq", "dq"], required=True)
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--ncalib", type=int, default=128)
    cli = ap.parse_args()
    torch.cuda.init()
    torch.zeros(1, device="cuda")

    t0 = time.perf_counter()
    model = common.load_bf16(cli.model)
    model.config.use_cache = False
    t_load = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()

    tm = Timer()
    if cli.method == "dq":
        run_dualquant(model, cli.model, tm)
    else:
        run_calibrated(cli.method, model, cli.model, cli.ncalib, tm)
    peak = torch.cuda.max_memory_allocated() / common.GB

    print(f"\n===== {cli.method} offline phase — {cli.model} (model load {t_load:.1f}s, excluded) =====")
    for k, v in tm.t.items():
        print(f"  {k:<30}{v:9.1f} s")
    print(f"RESULT {cli.method} offline total {sum(tm.t.values()):.1f} s  "
          f"peak GPU allocated {peak:.2f} GB")


if __name__ == "__main__":
    main()
