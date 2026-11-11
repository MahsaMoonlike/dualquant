# Dualquant codebase

Single-entry quantisation pipeline (`main.py`) with three orthogonal axes —
**method**, **weight format**, **activation format** — plus a **preprocess**
slot (SmoothQuant, QuaRot). The repo is self-contained: legacy helpers that
have not been rewritten in the new style are vendored byte-for-byte under
`legacy_vendor/` and exposed via `_legacy_path.py` (a tiny sys.path shim).

## Architecture (three orthogonal axes + preprocess)

| Axis            | What lives here                            | Selection                     |
|-----------------|--------------------------------------------|-------------------------------|
| Method          | `methods/` — rtn, dualquant, gptq, gptq_seq, awq, sinq | `--method` + `--method-cfg`   |
| Weight format   | `formats/` — mxfp4, mxint4, mxfp8\_e4m3, mxfp8\_e5m2, mxint8, nvfp4, sfp4, rtn\_int4, rtn\_int8 | `--weight-fmt --weight-block-size` |
| Activation fmt  | (same `formats/` registry; runs independently) | `--act-fmt --act-block-size --act-scale-format` |
| Scale format    | `scale_formats.py` — e8m0, e4m3, e4m4, e5m3, none | `--weight-scale-format` / `--act-scale-format` |
| Preprocess      | `preprocess/` — smoothquant, quarot         | `--preprocess` (hyperparams in `--method-cfg` JSON under same key) |

## Folder layout

```
dualquant/
├── main.py                      # single entrypoint (CLI flags drive everything)
├── run_sweep.sh                 # bash sweep over MODELS × METHODS × WEIGHT_FMTS
├── _legacy_path.py              # sys.path shim → legacy_vendor/
├── layer_wrapper_dualquant.py   # dualquant alpha/beta wrapper
├── scale_formats.py             # quantize_scale_e8m0/e4m3/e4m4/e5m3/none
├── formats/                     # FormatSpec registry (mxfp4, mxint4, mxfp8_*, nvfp4, sfp4, rtn_int{4,8})
├── methods/                     # rtn / dualquant / gptq / gptq_seq / awq / sinq
├── preprocess/                  # smoothquant.py, quarot.py
├── activations/                 # PermLinear runtime activation-quant wrapper
├── data/                        # calibration dataset + activation collection
├── quarot/                      # vendored spcl/QuaRot bits + R1/R2/R4 orchestrator
│   ├── llama_quarot.py          #   apply_r1 (R1 + optional R2 + R4 offline halves)
│   ├── online_had_install.py    #   install_online_hadamards (R2/R4 online halves)
│   └── vendor/                  #   byte-for-byte from github.com/spcl/QuaRot
├── smoothquant/
│   └── act_scales/              # per-model .pt activation statistics (used by --preprocess smoothquant)
├── legacy_vendor/               # byte-for-byte copies of unmigrated legacy modules
│   ├── awq.py, datautils.py, eval_utils.py, sinkhorn.py, torch_quant.py
│   ├── sinq_functions.py, layer_wrapper_baseline_data_formats_methods.py
│   └── smoothquant/             #   smooth_with_scale_dict.smooth_lm
├── blockfmt/                    # indirection to the block-format library (see Requirements)
├── configs/methods.json         # one merged file with per-method hyperparams
├── results/                     # created at runtime: per-model PPL CSVs from run_sweep.sh
├── scales_dualquant_rtn_int4/   # saved dualquant column scales (.pt)
├── scales_smoothquant_rtn_int4/ # saved SmoothQuant scales (.pt)
├── vendor/                      # third-party code kept in-tree to stay self-contained
│   └── dualscale_kernel_benchmark/  # fused_dual_scale_kernel used by the INT4 validators
├── ablation/                    # ablation studies: init, scale comparison, SQNR, INT4 kernel,
│                                #   real packed NVFP4 W4A4 path (real_nvfp4_kernel/, Blackwell)
└── README.md
```

Not tracked (regenerable): `hessian_cache/`, `scale_cache/`,
`ablation/real_int4_kernel/wrap_cache/`, `ablation/real_nvfp4_kernel/wrap_cache/`, and all measured outputs — the result
CSVs, plots and logs that `main.py`, `run_sweep.sh` and the `ablation/` scripts
write. The code and the plotting scripts are included so every number in the
paper can be regenerated; the recorded outputs themselves are not shipped.

## Requirements

Beyond `requirements.txt`, the pipeline needs an external block-format
quantization library that provides a `Format` class with
`Format.from_shorthand(...)` and `.cast(tensor)`. It is not publicly
redistributable, so it is resolved at import time from an environment
variable rather than being named in the source:

```bash
export BLOCKFMT_BACKEND=<python.module.path>   # module exposing `Format`
```

`blockfmt/__init__.py` raises a descriptive `ImportError` if it is unset.
Every weight/activation format goes through it, so nothing runs without it.

## Usage

All method/preprocess hyperparameters live in a single
`configs/methods.json` keyed by method name. Pass it once with
`--method-cfg`; main.py picks the right section automatically.

```bash
# RTN (no calibration, no hyperparams)
python main.py \
    --model Qwen/Qwen3-0.6B \
    --method rtn \
    --weight-fmt rtn_int4 --weight-block-size 128 --weight-scale-format none \
    --tasks wikitext

# Dualquant
python main.py \
    --model Qwen/Qwen3-0.6B \
    --method dualquant --method-cfg configs/methods.json \
    --weight-fmt mxfp4 --weight-block-size 32 --weight-scale-format e8m0 \
    --tasks wikitext --nsamples 128 --seqlen 2048

# GPTQ (collects calibration activations automatically)
python main.py \
    --model Qwen/Qwen3-0.6B \
    --method gptq --method-cfg configs/methods.json \
    --weight-fmt mxfp4 --weight-block-size 32 --weight-scale-format e8m0 \
    --tasks wikitext

# AWQ
python main.py \
    --model Qwen/Qwen3-0.6B \
    --method awq \
    --weight-fmt mxint4 --weight-block-size 32 --weight-scale-format e8m0 \
    --tasks wikitext

# SmoothQuant + GPTQ (preprocess composes with any method).
# --sq-scales overrides the JSON's act_scales_path; the bash sweep populates
# this per-model from smoothquant/act_scales/. For direct CLI use, point
# --sq-scales at a per-model .pt file in that directory.
python main.py \
    --model meta-llama/Llama-3.1-8B \
    --preprocess smoothquant --method-cfg configs/methods.json \
    --sq-scales smoothquant/act_scales/Llama-3.1-8B_seq_len_8192_4bit.pt \
    --method gptq \
    --weight-fmt mxfp4 --weight-block-size 32 --weight-scale-format e8m0 \
    --tasks wikitext

# QuaRot (R1; R2+R4 fire automatically when --act-quant on).
# Add a "quarot" block to configs/methods.json to pick rotate_mode/seed;
# defaults are rotate_mode=random, no seed.
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --preprocess quarot --method-cfg configs/methods.json \
    --method rtn \
    --weight-fmt rtn_int4 --weight-block-size 64 --weight-scale-format none \
    --act-quant on --act-fmt rtn_int4 --act-block-size 64 --act-scale-format none \
    --tasks wikitext

# Weight + activation co-quantisation (W4A8)
#   - rtn weights in mxfp4, activations in mxfp8_e4m3
python main.py \
    --model Qwen/Qwen3-0.6B \
    --method rtn \
    --weight-fmt mxfp4 --weight-block-size 32 --weight-scale-format e8m0 \
    --act-quant on --act-fmt mxfp8_e4m3 --act-block-size 32 --act-scale-format e8m0 \
    --tasks wikitext

# Dualquant with column-scale pre-scaling on activations (the per-col 1/beta
# returned by dualquant is fed into PermLinear.set_act_scale automatically)
python main.py \
    --model Qwen/Qwen3-0.6B \
    --method dualquant --method-cfg configs/methods.json \
    --weight-fmt mxfp4 --weight-block-size 32 --weight-scale-format e8m0 \
    --act-quant on --act-fmt mxfp8_e4m3 --act-block-size 32 --act-scale-format e8m0 \
    --act-scaled-before-quant \
    --tasks wikitext
```

To skip a layer family: `--no-attn-layers` or `--no-mlp-layers`.

## Sweeps

`run_sweep.sh` iterates over `MODELS × METHODS × WEIGHT_FMTS` (comment out
items in each array to skip them) and appends results to per-model CSVs
under `results/`. Environment overrides:

- `PREPROCESS=quarot bash run_sweep.sh` — applies QuaRot preprocess
- `PREPROCESS=smoothquant bash run_sweep.sh` — applies SmoothQuant; the
  script picks the right `.pt` file from `smoothquant/act_scales/` per
  model via `sq_scales_for()`. To add a new model, add a `case` entry there.
- `ACT_QUANT=off bash run_sweep.sh` — weight-only

## Vendored legacy modules

`legacy_vendor/` holds byte-for-byte copies of the legacy modules the new
code still consumes. `_legacy_path.py` puts this directory on `sys.path`.
The intent is that the repo runs standalone (no sibling-folder dependency):

- `awq.py`, `sinkhorn.py`, `sinq_functions.py`, `torch_quant.py`,
  `layer_wrapper_baseline_data_formats_methods.py`
- `datautils.py`, `eval_utils.py`
- `smoothquant/smooth_with_scale_dict.py` (the `smooth_lm` entry point)

Migrating any of these into the new style means: rewrite the relevant
functions under `methods/` / `data/` / `preprocess/`, remove the
corresponding `import _legacy_path` / legacy symbol, and delete the file
from `legacy_vendor/`. The QuaRot vendored package under `quarot/vendor/`
is structurally similar — byte-for-byte from `spcl/QuaRot`, deliberately
kept verbatim for parity with the upstream paper.

## Debug-only flag

`--debug-post-preprocess-ppl` evaluates PPL right after preprocess but
*before* quantisation. Useful for verifying that R1 (QuaRot) is FP-lossless
or that SmoothQuant moved scales correctly; not for normal runs. When
`--preprocess quarot --act-quant on`, this path also installs PermLinears
with `act_quant_enabled=False` plus the online Hadamards, so the eval
measures R1+R2+R4 cancellation in isolation.

## Config schema vs. legacy

The new schema has **no** `mse_reduction`/`split`/`smoothquant`/`awq`/`gptq`
boolean flags in JSON. Method/preprocess choice is a CLI flag. Hyperparameters
for every method and preprocess live in a single `configs/methods.json` keyed
by name:

```json
{
    "dualquant":   {"scale_option": "row_column", "num_iter": 15, ...},
    "gptq":        {"blocksize": 128, "percdamp": 0.01},
    "gptq_seq":    {"blocksize": 64, "percdamp": 0.01},
    "awq":         {},
    "sinq":        {},
    "rtn":         {},
    "smoothquant": {"alpha": 0.5},
    "quarot":      {"rotate_mode": "random", "seed": 0}
}
```

For SmoothQuant, the per-model `act_scales_path` is supplied at run time
via `--sq-scales` (the bash sweep populates it from `smoothquant/act_scales/`),
not from JSON. For QuaRot, `rotate_mode` is `"random"` (QR of Gaussian)
or `"hadamard"` (randomized Hadamard — needs `hidden_size` to be a power
of 2 or one of the special sizes in `quarot/vendor/hadamard_utils.get_hadK`).

main.py loads the JSON once and passes the relevant section to the active
method/preprocess. `scale_format` is part of the weight/activation format
CLI flags (`--weight-scale-format`, `--act-scale-format`), not the method
JSON, because it is a property of the format, not the optimisation method.
