"""Shared setup for the NVFP4 scripts: sys.path, backend selection, model id, PPL, prompt."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (ROOT, os.path.join(ROOT, "legacy_vendor"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
# The repo's block-format backend, provided here by nvfp4_numerics (see its docstring).
os.environ.setdefault("BLOCKFMT_BACKEND", "nvfp4_numerics")

import torch
import torch.nn.functional as F

# meta-llama/Llama-3.1-8B is gated; this ungated mirror has the identical weights.
DEFAULT_MODEL = os.environ.get("DQ_MODEL", "unsloth/Meta-Llama-3.1-8B")
PROMPT = "The quick brown fox jumps over the lazy dog. " * 8      # 82 tokens, as in the paper
GB = 1e9
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def iter_linears(model):
    """(layer_idx, parent, attr, key) for the 224 quantized projections."""
    for i, blk in enumerate(model.model.layers):
        for n in ATTN:
            yield i, blk.self_attn, n, f"layer{i}.self_attn.{n}"
        for n in MLP:
            yield i, blk.mlp, n, f"layer{i}.mlp.{n}"


def load_bf16(model_id, dev=None):
    from main import load_model
    m = load_model(model_id)
    m.config.use_cache = True
    if dev is not None:
        m = m.to(dev)
    return m.eval()


def wikitext_test_ids(model_id, seqlen):
    import datautils
    _, _, _, testenc, _ = datautils.get_loaders("wikitext", model_id, nsamples=128, seed=0,
                                                seqlen=seqlen)
    return testenc.input_ids if hasattr(testenc, "input_ids") else testenc


LEAN_HEAD = os.environ.get("DQ_LEAN_HEAD", "0") == "1"


@torch.no_grad()
def ppl(model, ids, seqlen, dev, chunk=256, log_every=20):
    """Non-overlapping windows, fp32 NLL accumulated in slices (same as
    ../real_int4_kernel/ppl_eval.py, so numbers are comparable to the INT4 path).

    DQ_LEAN_HEAD=1 (for models that barely fit, e.g. Qwen3-14B BF16 on 32 GB): run the decoder
    once, then apply lm_head per `chunk` rows instead of materializing the full
    [seqlen, vocab] logits (2.5 GB for Qwen3 at 8192). Same math; only the lm_head GEMM's M
    differs, so results can differ from the default path in the last bits."""
    n = ids.numel() // seqlen
    nll, tok = 0.0, 0
    for i in range(n):
        w = ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        if LEAN_HEAD:
            h = model.model(w, use_cache=False).last_hidden_state[0, :-1]
            tgt = w[0, 1:]
            for s in range(0, h.shape[0], chunk):
                lg = model.lm_head(h[s:s + chunk]).float()
                nll += F.cross_entropy(lg, tgt[s:s + chunk], reduction="sum").item()
            tok += tgt.numel()
            del h
            if log_every and (i + 1) % log_every == 0:
                print(f"    window {i+1}/{n}  running PPL={torch.tensor(nll/tok).exp():.4f}", flush=True)
            continue
        logits = model(w, use_cache=False).logits
        src, tgt = logits[0, :-1], w[0, 1:]
        for s in range(0, src.shape[0], chunk):
            nll += F.cross_entropy(src[s:s + chunk].float(), tgt[s:s + chunk],
                                   reduction="sum").item()
        tok += tgt.numel()
        del logits, src
        if log_every and (i + 1) % log_every == 0:
            print(f"    window {i+1}/{n}  running PPL={torch.tensor(nll/tok).exp():.4f}", flush=True)
    return float(torch.tensor(nll / tok).exp())
