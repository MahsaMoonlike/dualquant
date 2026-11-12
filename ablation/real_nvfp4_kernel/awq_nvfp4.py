"""AWQ+ for NVFP4 W4A4: the repo's AWQ scale search, run block by block, plus a disk cache.

The paper's AWQ+ baseline (methods/awq.py) searches, per linear and independently, a
per-input-channel scale s = mu_x^a / mu_w^(1-a) over a 20-point grid of a, minimising
||X W^T - X (Q(W s)/s)^T||, then folds s back into the weight: W_eff = Q(W s) / s, with
activations RTN-quantized unscaled. That fake-quant form is not executable on packed NVFP4
hardware (the per-column 1/s is not part of the format). The exact deployable form is:
store Q(W s) packed and apply 1/s to the activation before quantizing it, which is the same
fused per-channel multiply DualQuant uses for beta. Because the scales are per layer (not
shared across q/k/v or gate/up), they cannot be folded into the RMSNorms as SmoothQuant's are.

  method "awq"        -> (mat_q = Q_nvfp4(W s), col = 1/s)      deployable, real path
  fake "awq_paper"    -> W_eff = Q_nvfp4(W s)/s, activations unscaled  (methods/awq.py form)

Calibration is the repo's own (legacy awq.get_calib_dataset: WikiText-2 train, 512 samples,
512-token chunks; scales from the BF16 model's activations, as main.py does). The repo
collects every linear's input for the whole set at once (~90 GB for an 8B model); here the
model is run block by block and only one block's inputs are held at a time.

Usage (populates the cache; ~a few minutes):  python awq_nvfp4.py [--model ...]
"""
import hashlib
import os

import common
import torch

import nvfp4_numerics as N

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrap_cache")
# Calibration-set size. Unset = the repo's collect_calib_activations setting (512 WikiText-2
# lines -> 512-token chunks, first 128 kept; only ~93 exist for Llama-3.1-8B). An integer n
# takes exactly n chunks from the same generator (calib_cost.py uses 128).
NCALIB = int(os.environ["DQ_AWQ_NCALIB"]) if os.environ.get("DQ_AWQ_NCALIB") else None


def _cache_path(model_id, key, ncalib=None):
    ncalib = NCALIB if ncalib is None else ncalib
    sfx = "" if ncalib is None else f"_n{ncalib}"
    tag = f"{model_id}|{key}|awq|nvfp4|{N.MODE}" + sfx
    h = hashlib.sha1(tag.encode()).hexdigest()[:8]
    return os.path.join(_CACHE_DIR,
                        f"{model_id.replace('/', '__')}__{key}__awq_nvfp4_{N.MODE}{sfx}__{h}.pt")


def calib_set(tok, ncalib=None):
    """The repo's AWQ calibration chunks (legacy awq.get_calib_dataset: WikiText-2 train lines
    of <=512 tokens, concatenated, split into 512-token chunks)."""
    from awq import get_calib_dataset
    if ncalib is None:
        return get_calib_dataset(tokenizer=tok, n_samples=512, max_seq_len=512)[:128]
    lines = 512
    while True:
        c = get_calib_dataset(tokenizer=tok, n_samples=lines, max_seq_len=512)
        if len(c) >= ncalib:
            return c[:ncalib]
        lines *= 2


def all_cached(model_id, model_cfg):
    keys = [f"layer{i}.{p}.{n}" for i in range(model_cfg.num_hidden_layers)
            for p, ns in (("self_attn", common.ATTN), ("mlp", common.MLP)) for n in ns]
    return all(os.path.exists(_cache_path(model_id, k)) for k in keys)


@torch.no_grad()
def compute_awq(model_id, dev="cuda"):
    """Run the AWQ search for every projection and cache {mat_q, col, s}.

    Block-sequential (GPTQ-style): the calibration hidden states (~0.4 GB) stay on the GPU and
    only the current decoder block is moved there, so the repo's search (several [tokens, N]
    fp32 temporaries) has room. Each block is called exactly as the model calls it (no mask ->
    SDPA is_causal, shared rotary embeddings), and is the unquantized BF16 block, so the
    captured inputs are the BF16 model's activations, as in main.py's AWQ path."""
    import transformers
    from awq import compute_awq_scale                              # legacy_vendor (repo's)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    model = common.load_bf16(model_id)
    model.config.use_cache = False
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    calib = calib_set(tok, NCALIB)
    print(f"AWQ calibration: {len(calib)} x {calib[0].shape[1]} tokens", flush=True)
    fq = lambda w: N.nvfp4_fake_quant(w.to(dev))

    model.model.embed_tokens.to(dev)
    model.model.rotary_emb.to(dev)
    hs = [model.model.embed_tokens(x.to(dev)) for x in calib]
    pos = torch.arange(hs[0].shape[1], device=dev)[None]
    pe = model.model.rotary_emb(hs[0], pos)

    for i, blk in enumerate(model.model.layers):
        names = [("self_attn", n) for n in common.ATTN] + [("mlp", n) for n in common.MLP]
        done = all(os.path.exists(_cache_path(model_id, f"layer{i}.{p}.{n}")) for p, n in names)
        blk.to(dev)
        acts = {n: [] for _, n in names}
        hooks = [] if done else [getattr(getattr(blk, p), n).register_forward_hook(
            lambda m, inp, out, n=n: acts[n].append(inp[0].detach())) for p, n in names]
        for j in range(len(hs)):
            out = blk(hs[j], attention_mask=None, position_ids=pos, position_embeddings=pe)
            hs[j] = out[0] if isinstance(out, tuple) else out
        for h in hooks:
            h.remove()
        if not done:
            for p, n in names:
                lin = getattr(getattr(blk, p), n)
                W = lin.weight.data.float()
                X = torch.cat(acts[n], 0)                           # [chunks, 512, K] bf16
                acts[n] = None
                s = compute_awq_scale(W, X, fq).float()             # repo's search, on GPU
                mat_q = N.nvfp4_fake_quant(W * s[None, :]).to(torch.bfloat16)
                torch.save({"mat_q": mat_q.cpu(), "col": (1.0 / s).cpu(), "s": s.cpu()},
                           _cache_path(model_id, f"layer{i}.{p}.{n}"))
                del X, W
                torch.cuda.empty_cache()
            print(f"  AWQ scales for layer {i} done", flush=True)
        blk.cpu()
        del acts
        torch.cuda.empty_cache()
    del model, hs
    torch.cuda.empty_cache()


def load_awq(model_id, key):
    d = torch.load(_cache_path(model_id, key), map_location="cpu")
    return d["mat_q"], d["col"], d["s"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    compute_awq(ap.parse_args().model)
