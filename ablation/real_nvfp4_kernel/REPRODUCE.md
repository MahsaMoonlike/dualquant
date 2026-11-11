# Real packed NVFP4 W4A4 path (DualQuant, Llama-3.1-8B, RTX 5090)

The NVFP4 counterpart of `../real_int4_kernel/`. It produces the same three tables as the paper's
Table 3 (memory + throughput), Table 11 (per-component GPU time) and Table 12 (memory breakdown),
plus the real-vs-fake WikiText-2 check, for NVFP4 (G=16, E4M3 block scales) on a Blackwell GPU.
Exact environment: `ENVIRONMENT.txt` (RTX 5090 sm_120, driver 580, torch 2.14.1+cu130,
triton 3.8, torchao 0.18, transformers 5.15).

Unlike the INT4 path, nothing is built from source: the GEMM is cuBLASLt's native block-scaled
FP4 GEMM (`torch.nn.functional.scaled_mm`), and the activation quantizer is a Triton kernel.

## 1. Environment

```bash
sudo apt-get install -y python3.12-venv            # if ensurepip is missing
python3 -m venv ~/venvs/dq-nvfp4 && source ~/venvs/dq-nvfp4/bin/activate
pip install -U pip uv
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
echo "torch==2.14.1+cu130" > /tmp/c.txt             # keep later installs from moving torch
uv pip install -c /tmp/c.txt --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
    -r ablation/real_nvfp4_kernel/requirements.txt
```

`pyarrow<19`: pyarrow 19+ requires NumPy 2, and the repo's legacy code pins NumPy 1.26.

Model: `meta-llama/Llama-3.1-8B` is gated; the scripts default to the ungated mirror
`unsloth/Meta-Llama-3.1-8B` (identical weights). Override with `DQ_MODEL=...`.

## 2. The missing numerics library: `nvfp4_numerics.py`

The repo routes every block format through `$BLOCKFMT_BACKEND` (not redistributable). Here
`nvfp4_numerics.py` provides that interface for NVFP4 only (`Format.from_shorthand("NVFP4[E2M1]{16}")`),
with numerics taken op for op from torchao's `nvfp4_quantize`, i.e. the definition cuBLAS
consumes. It is checked bit-exact against torchao, and its FP4 rounding against the repo's
own `torch_quant.float_to_fp4`. Every script sets `BLOCKFMT_BACKEND=nvfp4_numerics` itself.

**Single- vs two-level.** NVFP4 can carry a per-tensor FP32 scale on top of the E4M3 block
scales. Which one the paper used is decided empirically, with fake-quant RTN W4A4 on Llama-3.1-8B
(WikiText-2, seqlen 8192):

| NVFP4 variant | RTN W4A4 PPL | paper Table 1 |
|---|---|---|
| single-level (E4M3 block scales only) | **9.03** | **9.04** |
| two-level (+ dynamic per-tensor scale) | 6.23 | – |

So the default is `NVFP4_TENSOR_SCALE=single_level`. This matters for the method: single-level
E4M3 cannot represent the small block scales of raw weights (RTN weight SQNR 14.5–17.9 dB),
and DualQuant's row/column scales fix exactly that (18.7–19.5 dB). Under two-level, RTN is
already at ~20.5–21 dB and DualQuant does not improve weight MSE.

## 3. What the real path does

| Piece | Implementation |
|---|---|
| DualQuant BCD | `dq_nvfp4_bcd.py`: the repo's `wrap_dualquant_layer` algorithm (`configs/methods.json` settings, scale format E4M3) batched over all 16-column blocks on the GPU. On CPU it is **bit-identical** to the repo's loop; on GPU float-summation order differs, with equal reconstruction MSE. ~1 min for all 224 projections, cached in `wrap_cache/`. `DQ_BCD_DEVICE=cpu` gives the bit-identical (slow) variant. |
| Weights | FP4 E2M1 packed 2/byte + E4M3 scale per 16 + FP32 per-tensor scale, scales pre-swizzled to cuBLAS's 32_4_4 layout. Never dequantized at runtime. |
| Activations | `dq_nvfp4_quantize_act_kernel` (Triton): loads x, multiplies by DualQuant's column scale, computes the per-16 E4M3 scale, rounds to FP4, packs, and writes scales directly in swizzled layout. **One launch per linear, beta included.** Bit-exact with the fake-quant reference (`verification/test_act_quant_bitexact.py`). |
| GEMM | `F.scaled_mm` → cuBLASLt block-scaled FP4 GEMM, FP32 accumulate, BF16 out; the weight's per-tensor scale is passed as a TensorWise scale (folded into alpha, no extra kernel). |
| Rest | embedding, lm_head, RMSNorm, RoPE, attention, KV cache in BF16. |

**Why real ≠ fake exactly (and why INT4 was closer).** DualQuant's NVFP4 weight block is
`alpha_inv[r,b] · s8[r,b] · fp4`: two stacked E4M3 scales (the row scale and the inner
block-format scale). The hardware has one E4M3 scale per block, and a product of two E4M3
numbers is generally not E4M3, so packing `mat_q` is not lossless. (INT4's scales were BF16,
8 mantissa bits, so its re-packing was essentially exact: 6.74 vs 6.73.) The packer keeps the
loss small: a power-of-two per-tensor scale makes every block whose stacked scale *is*
representable pack exactly (~65% of elements), and the rest pick the MSE-best of the absmax
E4M3 scale and its ±2 neighbours. Weight SQNR (vs original W) drops by ~0.4–0.5 dB.

## 4. Commands

```bash
cd ablation/real_nvfp4_kernel
./run_all.sh           # everything below, logs in logs/<model>/
DQ_MODEL=Qwen/Qwen3-14B DQ_LEAN_HEAD=1 ./run_all.sh   # another model (lean lm_head for PPL)
```

or individually (benchmarks: run twice, report the warm second run):

```bash
python verification/test_act_quant_bitexact.py      # kernel == reference, bit for bit
python verification/test_bcd_matches_reference.py   # batched BCD vs repo loop
python fake_quant_ppl.py --method dualquant         # fake-quant W4A4 PPL (Table 1 analogue)
python ppl_eval.py --mode nvfp4                     # real packed W4A4 PPL
python graph_bench_all.py --mode bf16|nvfp4         # Table 3 throughput
python profile_breakdown.py --mode bf16|nvfp4       # Table 11
python memory_breakdown.py --mode bf16|nvfp4        # Table 12 / Table 3 memory
```

## 5. Harness notes (transformers 5.x)

- `StaticCache` now keeps its write position in a per-layer **device** counter
  (`cumulative_length`) and ignores `cache_position`. The decode step is therefore made fully
  device-driven (it feeds back its own argmax and advances `cache_position`), so each replay is
  one real autoregressive step and the 100 timed replays grow the cache 82 → 182, as in the
  paper. Without this the captured graph silently decodes garbage.
- While a CUDA stream is capturing, HF materializes an explicit causal mask and disables
  SDPA's GQA path, so captured prefill runs a different attention kernel than eager. For an
  unpadded, cache-free prompt `is_causal=True` is exact, so `graph_bench_all.py` routes that
  case to it in both modes (BF16 and NVFP4 alike). With that, captured prefill logits are
  bit-identical to eager.
- Everything is torch-allocated, so (unlike nunchaku) `torch.cuda.memory_allocated()` sees
  the quantized weights; `mem_get_info` is still used for the resident totals.

## 6. Results — Llama-3.1-8B (RTX 5090, logs/Meta-Llama-3.1-8B/, warm second runs)

WikiText-2 PPL (seqlen 8192): BF16 5.609 · fake RTN W4A4 9.026 (paper 9.04) ·
fake DualQuant W4A4 6.693 (paper 6.83) · **real packed DualQuant W4A4 6.736**.

| Workload | Mem BF16 | Mem W4A4 | Reduction | BF16 tok/s | W4A4 tok/s | Speedup |
|---|---|---|---|---|---|---|
| Prefill b=1  | 16.30 | 6.20 | 2.63× | 5702.1  | 9093.4  | 1.59× |
| Prefill b=16 | 16.62 | 6.53 | 2.55× | 12790.6 | 35785.7 | 2.80× |
| Decode b=1   | 16.30 | 6.21 | 2.63× | 84.3    | 103.6   | 1.23× |
| Decode b=16  | 16.70 | 6.61 | 2.53× | 901.7   | 1262.5  | 1.40× |

Decode b=1 is limited by cuBLASLt's FP4 kernel at M=1 (128x128 tiles, no split-K: ~0.7 TB/s
effective vs ~1.6 TB/s for BF16 cuBLAS). A Triton small-M W4A4 kernel was tried and was 3x
slower than cuBLAS, so it was dropped.

## 7. Results — Qwen3-14B (RTX 5090, logs/Qwen3-14B/, warm second runs)

`DQ_MODEL=Qwen/Qwen3-14B DQ_LEAN_HEAD=1 ./run_all.sh`. Prompt is 81 tokens (no BOS).

| Workload | Mem BF16 | Mem W4A4 | Reduction | BF16 tok/s | W4A4 tok/s | Speedup |
|---|---|---|---|---|---|---|
| Prefill b=1  | 29.64 | 10.96 | 2.70× | 3067.4 | 5542.9  | 1.81× |
| Prefill b=16 | 30.02 | 11.34 | 2.65× | 6790.3 | 20512.5 | 3.02× |
| Decode b=1   | 29.65 | 10.97 | 2.70× | 47.5   | 66.0    | 1.39× |
| Decode b=16  | 30.14 | 11.46 | 2.63× | 555.7  | 824.1   | 1.48× |

WikiText-2 PPL (seqlen 8192): BF16 7.515 (paper 7.54) · fake RTN W4A4 7.906 (paper 7.93) ·
DualQuant W4A4 fake 9.290 / real 9.260 · with `DQ_BETA_GAUGE=median` fake 8.145 / real 8.127
(paper 7.75).

**Known issue (Qwen3).** DualQuant's column scale reaches ~750x on Qwen3-14B, so x*col
saturates single-level NVFP4's E4M3 block scale (448, i.e. block absmax 2688) in 60/280
linears. `DQ_BETA_GAUGE=median` (dq_nvfp4_bcd.apply_gauge) divides col by a power of two and
multiplies mat_q by it: exact (all FP4 codes and E4M3 scales unchanged, calibration-free) and
recovers most of the loss, but the result is still above RTN and the paper. It is off by
default, so the Llama numbers in section 6 are without it. Throughput and memory are unaffected.
