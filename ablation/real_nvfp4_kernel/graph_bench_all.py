"""Throughput for prefill/decode x batch 1/16, BF16 vs real packed NVFP4 W4A4 (Table-3 style).

Same methodology as ../real_int4_kernel/graph_bench_all.py: every configuration, BF16
included, is CUDA-graph captured so the comparison measures GPU work rather than Python
dispatch, and every graph is checked against the uncaptured execution before it is timed
(prefill: logits bit-exact; decode: generated tokens identical over the validation steps).

  prefill : 82-token prompt, no KV cache, one forward per replay
  decode  : one StaticCache decode step per replay (prompt 82 + 100 generated = 182-token
            cache, as in the paper), timed over 100 replays

Usage:
    python graph_bench_all.py --mode bf16
    python graph_bench_all.py --mode nvfp4
"""
import argparse
import time

import common
import torch
import transformers
from transformers import StaticCache

from full_model_nvfp4 import build_model


def use_causal_flag_without_padding():
    """Make eager and CUDA-graph-captured prefill run the same attention kernel.

    In eager mode HF (transformers 5.x) calls SDPA with is_causal=True, enable_gqa=True for an
    unpadded prompt. While a CUDA stream is capturing it cannot inspect the padding mask, so
    it materializes an explicit [q, kv] mask and expands K/V instead -- a different kernel,
    so captured logits would differ from eager ones and the bit-exact check would be
    meaningless (and the timed graph would not be the eager computation).
    For our prefill (one unpadded prompt, no KV cache, q_len == kv_len) the causal flag is
    exact, so route that case to SDPA(is_causal=True, enable_gqa=True) in both modes.
    Decode steps (q_len == 1 against a StaticCache) are left to HF untouched. Applied
    identically to the BF16 and NVFP4 models."""
    import torch.nn.functional as F
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
             is_causal=None, **kw):
        if query.shape[2] > 1 and query.shape[2] == key.shape[2] and dropout == 0.0:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                                                 is_causal=True, scale=scaling, enable_gqa=True)
            return out.transpose(1, 2).contiguous(), None
        return orig(module, query, key, value, attention_mask, dropout=dropout,
                    scaling=scaling, is_causal=is_causal, **kw)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa


use_causal_flag_without_padding()


def timed_replay(g, n, per_call_tokens, warm=5):
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    return per_call_tokens / dt, dt * 1e3


def _warm_on_side_stream(fn, n=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()


@torch.no_grad()
def capture_prefill(model, ids, bs):
    ids_b = ids.repeat(bs, 1)
    _warm_on_side_stream(lambda: model(ids_b, use_cache=False))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model(ids_b, use_cache=False).logits
    return g, out, ids_b


@torch.no_grad()
def prefill(model, ids, bs, reps=20):
    ref = model(ids.repeat(bs, 1), use_cache=False).logits.float().clone()
    g, out, ids_b = capture_prefill(model, ids, bs)
    g.replay(); torch.cuda.synchronize()
    ok = torch.equal(out.float(), ref)
    diff = (out.float() - ref).abs().max().item()
    tps, ms = timed_replay(g, reps, bs * ids_b.shape[1])
    return tps, ms, ok, diff


class DecodeStep:
    """A StaticCache holding the prompt, plus one self-advancing decode step.

    transformers 5.x StaticCache keeps its write position in a per-layer DEVICE counter
    (StaticLayer.cumulative_length, advanced with add_ on every update) and no longer takes
    it from cache_position. So the step is made fully device-driven: it writes the argmax
    token back into its own input and bumps cache_position by one. A captured replay is then
    one real autoregressive step, and N replays generate N tokens (prompt 82 -> 182 for the
    timed 100), exactly like generate(). reset() rewinds all of it to just after the prompt.
    """

    def __init__(self, model, ids, bs, max_len, dev):
        self.model, self.plen = model, ids.shape[1]
        ids_b = ids.repeat(bs, 1)
        self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        pos = torch.arange(self.plen, device=dev)
        o = model(ids_b, past_key_values=self.cache, cache_position=pos, use_cache=True)
        self.si = o.logits[:, -1:].argmax(-1).clone()
        self.first = self.si.clone()
        self.sp = torch.tensor([self.plen], device=dev, dtype=torch.long)

    def __call__(self):
        logits = self.model(input_ids=self.si, past_key_values=self.cache,
                            cache_position=self.sp, use_cache=True).logits
        self.si.copy_(logits[:, -1:].argmax(-1))
        self.sp.add_(1)
        return logits

    def reset(self, pos=None):
        """Rewind to just after the prompt (or to `pos`, for profiling at a given cache length;
        cache slots beyond the prompt then hold earlier steps' K/V, which is fine for timing)."""
        pos = self.plen if pos is None else pos
        self.si.copy_(self.first)
        self.sp.fill_(pos)
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(pos)


@torch.no_grad()
def decode(model, ids, bs, dev, new_tokens=100, val_steps=8):
    max_len = ids.shape[1] + new_tokens + 8

    # reference: uncaptured stepping
    st = DecodeStep(model, ids, bs, max_len, dev)
    ref = []
    for _ in range(val_steps):
        st()
        ref.append(st.si[:, 0].tolist())

    # captured, on a fresh cache
    st = DecodeStep(model, ids, bs, max_len, dev)
    _warm_on_side_stream(st)
    st.reset()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        st()
    st.reset()
    got = []
    for _ in range(val_steps):
        g.replay()
        got.append(st.si[:, 0].tolist())
    ok = got == ref
    st.reset()
    tps, ms = timed_replay(g, new_tokens, bs, warm=0)   # 100 real steps, cache 82 -> 182
    return tps, ms, ok, (ref, got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--mode", choices=["bf16", "nvfp4", "sq_nvfp4", "awq_nvfp4"], required=True)
    cli = ap.parse_args()
    dev = torch.device("cuda:0")
    model = build_model(cli.model, cli.mode, dev)
    tok = transformers.AutoTokenizer.from_pretrained(cli.model)
    ids = tok(common.PROMPT, return_tensors="pt").input_ids.to(dev)
    print(f"\n===== {cli.mode.upper()} | prompt {ids.shape[1]} tokens | all configs CUDA-graph captured =====")
    for bs in (1, 16):
        tps, ms, ok, diff = prefill(model, ids, bs)
        print(f"PREFILL batch={bs:<3} {tps:9.1f} tok/s  {ms:8.3f} ms  logits_bit_exact={ok} (maxdiff={diff:g})")
    for bs in (1, 16):
        tps, ms, ok, info = decode(model, ids, bs, dev)
        print(f"DECODE  batch={bs:<3} {tps:9.1f} tok/s  {ms:8.3f} ms/step  tokens_match={ok}")
        if not ok:
            print(f"    ref={info[0]}\n    got={info[1]}")


if __name__ == "__main__":
    main()
