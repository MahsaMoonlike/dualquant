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

**Packing DualQuant's weights.** DualQuant's NVFP4 weight block is `alpha_inv[r,b] · s8[r,b] · fp4`:
two stacked E4M3 scales, while the hardware has one E4M3 scale per block (+ one FP32 per
tensor). The packer (`quantize_weight`) makes this near-lossless: a power-of-two per-tensor
scale, and per block the MSE-best of absmax/6, its ±2 E4M3 neighbours, and absmax/{4,3,2}
(the latter catch blocks whose largest FP4 code is not 6, which are the majority of the
non-exact ones). Result: 47.8–68.9 dB SQNR vs `mat_q`, 99.7–100% of elements exact, and real
PPL within 0.01 of fake quant (6.685 vs 6.693).

**SmoothQuant baseline (`--mode sq_nvfp4`).** The paper's SmoothQuant (alpha 0.5, the repo's
calibrated `smoothquant/act_scales/`, upstream recipe: q/k/v and gate/up smoothed and folded
into the RMSNorms, o/down unsmoothed), NVFP4 RTN weights, packed into the same
`NVFP4W4A4Linear` with no runtime column scale (kernel variant without the multiply).
Same formats, same GEMM: it isolates what DualQuant's per-layer beta costs at runtime.

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

Throughput for BF16, SmoothQuant and DualQuant from one session (logs/Meta-Llama-3.1-8B/graph_bench_*_run2.log);
AWQ+ from a later session (logs/Meta-Llama-3.1-8B/cmp/); all speedups vs that BF16. SmoothQuant: alpha 0.5,
repo's calibrated act scales, folded into RMSNorms (`sq_nvfp4`). AWQ+: repo's AWQ search
(`awq_nvfp4.py`), deployed as packed Q(W s) with 1/s fused into the activation quantizer
(`awq_nvfp4`), the same runtime mechanism as DualQuant's beta.

| | BF16 | SmoothQuant | AWQ+ | DualQuant |
|---|---|---|---|---|
| Calibration-free | – | no | no | yes |
| WikiText-2 PPL, real packed (fake) | 5.609 | 7.281 (7.277) | 22.406 (22.493) | **6.685** (6.693) |
| Paper Table 1 | 5.62 | 7.32 | 22.79 | 6.83 |
| Prefill b=1 tok/s | 5704.9 | 9145.8 (1.60×) | 9102.0 (1.60×) | 9100.6 (1.60×) |
| Prefill b=16 tok/s | 12822.1 | 36134.4 (2.82×) | 36352.1 (2.84×) | 36117.8 (2.82×) |
| Decode b=1 tok/s | 84.4 | 104.4 (1.24×) | 103.6 (1.23×) | 103.5 (1.23×) |
| Decode b=16 tok/s | 901.7 | 1268.9 (1.41×) | 1262.8 (1.40×) | 1262.7 (1.40×) |
| Memory prefill b=1 / b=16 (GB) | 16.30 / 16.62 | 6.18 / 6.51 | 6.20 / 6.53 | 6.20 / 6.53 |
| Memory decode b=1 / b=16 (GB) | 16.30 / 16.70 | 6.19 / 6.59 | 6.21 / 6.61 | 6.21 / 6.61 |

Activation quantization (ms, prefill b1 / b16 / decode b1 / b16): SmoothQuant 0.335 / 1.532 /
0.222 / 0.233; AWQ+ 0.380 / 1.632 / 0.291 / 0.306; DualQuant 0.379 / 1.634 / 0.291 / 0.305.
The fused per-channel multiply costs 0.04–0.10 ms per forward (<0.8% of GPU time).
Run-to-run throughput noise is ~0.5%. AWQ+ "fake" is the deployable form (x/s quantized);
the repo's methods/awq.py form (Q(W s)/s, x unscaled) gives 22.279.
Decode b=1 is limited by cuBLASLt's FP4 kernel at M=1 (128x128 tiles, no split-K).

### Offline cost: calibration (SmoothQuant, AWQ+) vs optimization (DualQuant)

`python calib_cost.py --method sq|awq|dq` (logs/Meta-Llama-3.1-8B/calib/, run 2 of 2; run 1 within
8%). Timed from the BF16 model in host memory (model load excluded for all), one decoder block
(sq, awq) or one linear (dq) on the GPU at a time.
- SmoothQuant: per-channel act absmax over 128 x 512 WikiText-2 train tokens (the repo's
  calibration chunks), alpha swept per smoothing point over the SmoothQuant paper's sweet spot
  {0.4, 0.5, 0.6} by NVFP4 W4A4 output error of the group's linears; chose 0.5 x35, 0.6 x28, 0.4 x1.
- AWQ+: the repo's procedure (compute_awq_scale per linear, 20-point grid, no clipping search)
  on the same 128 x 512 tokens.
- DualQuant: BCD with T = 15 (paper, Algorithm 1); output bit-identical to the cached scales
  the real path deploys (224/224 linears).

| | SmoothQuant (alpha sweep) | AWQ+ | DualQuant |
|---|---|---|---|
| Calibration data | 128 x 512 tokens | 128 x 512 tokens | none |
| Calibration data prep | 3.0 s | 3.0 s | – |
| Forward passes (block-sequential) | 11.8 s | 12.3 s | – |
| Search / optimization | 45.5 s (3 alphas x 64 points) | 359.9 s (20 grid x 224 linears) | 41.0 s (BCD, 224 linears) |
| **Total** | **60.3 s** (1.47x DQ) | **375.2 s** (9.15x DQ) | **41.0 s** |
| Peak GPU memory | 5.45 GB | 21.85 GB | 2.22 GB |
| WikiText-2 PPL, real packed | 7.334 | 18.510 | **6.685** |

PPL is for these 128-sample calibrations; with the repo's shipped SmoothQuant act scales
(alpha 0.5) SmoothQuant gives 7.281 and AWQ+ with the repo's ~93 chunks 22.406 (table above).

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

### Qwen3-14B: DualQuant vs SmoothQuant vs AWQ+ (logs/Qwen3-14B/cmp/, one session, run 2)

Same configuration as section 6 (SmoothQuant alpha 0.5 with the repo's
`Qwen3-14B_seq_len_8192_4bit.pt` act scales; AWQ+ calibrated by `awq_nvfp4.py`, 95 x 512 tokens).
DualQuant real PPL re-measured with the current packer.

| | BF16 | SmoothQuant | AWQ+ | DualQuant | DualQuant + gauge |
|---|---|---|---|---|---|
| Calibration-free | – | no | no | yes | yes |
| WikiText-2 PPL, real packed (fake) | 7.515 | **7.717** (7.722) | 8.055 (8.061) | 9.313 (9.290) | 8.157 (8.145) |
| Paper | 7.54 | – | – | 7.75 | 7.75 |
| Prefill b=1 tok/s | 3074.1 | 5574.7 (1.81×) | 5550.8 (1.81×) | 5550.8 (1.81×) | same |
| Prefill b=16 tok/s | 6871.4 | 20923.2 (3.04×) | 20963.9 (3.05×) | 21017.0 (3.06×) | same |
| Decode b=1 tok/s | 47.5 | 66.5 (1.40×) | 66.1 (1.39×) | 66.2 (1.39×) | same |
| Decode b=16 tok/s | 556.2 | 830.0 (1.49×) | 827.4 (1.49×) | 827.6 (1.49×) | same |
| Memory prefill b=1 / b=16 (GB) | 29.64 / 30.02 | 10.94 / 11.32 | 10.96 / 11.34 | 10.96 / 11.34 | same |
| Memory decode b=1 / b=16 (GB) | 29.65 / 30.14 | 10.95 / 11.44 | 10.97 / 11.46 | 10.97 / 11.46 | same |

Activation quantization (ms, prefill b1 / b16 / decode b1 / b16): SmoothQuant 0.441 / 2.448 /
0.276 / 0.289; AWQ+ 0.499 / 2.555 / 0.362 / 0.379; DualQuant 0.498 / 2.551 / 0.361 / 0.379.
Fake RTN W4A4 is 7.906 (paper 7.93); AWQ in the methods/awq.py form (Q(W s)/s) is 7.976. On Qwen3-14B both calibrated baselines beat DualQuant:
DualQuant's column scale (up to ~750) saturates single-level NVFP4 activations (see the known
issue above), whereas SmoothQuant's folded scale shrinks the outlier channels before quantization.

### Qwen3-14B offline cost: calibration (SmoothQuant, AWQ+) vs optimization (DualQuant)

Same procedure and settings as for Llama-3.1-8B in section 6 (logs/Qwen3-14B/calib/, run 2 of 2;
run 1 within 4%). SmoothQuant chose alpha 0.6 x68, 0.5 x12. DualQuant's BCD output is
bit-identical to the deployed cache (280/280 linears).

| | SmoothQuant (alpha sweep) | AWQ+ | DualQuant | DualQuant + gauge |
|---|---|---|---|---|
| Calibration data | 128 x 512 tokens | 128 x 512 tokens | none | none |
| Calibration data prep | 2.9 s | 2.8 s | – | – |
| Forward passes (block-sequential) | 25.0 s | 30.1 s | – | – |
| Search / optimization | 92.6 s (3 alphas x 80 points) | 744.2 s (20 grid x 280 linears) | 84.9 s (BCD, 280 linears) | same |
| **Total** | **120.5 s** (1.42x DQ) | **777.0 s** (9.15x DQ) | **84.9 s** | same |
| Peak GPU memory | 7.64 GB | 27.40 GB | 3.37 GB | same |
| WikiText-2 PPL, real packed | **7.804** | 8.087 | 9.313 | 8.157 |

With the repo's shipped SmoothQuant act scales (alpha 0.5) SmoothQuant gives 7.717 and AWQ+
with ~95 chunks 8.055 (table above). The gauge is a power-of-two rescale of beta (no extra cost).
