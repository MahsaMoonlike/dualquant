#!/bin/bash
# Regenerates every NVFP4 number (accuracy, throughput, per-component time, memory) into
# logs/<model name>/.  Model: DQ_MODEL (default unsloth/Meta-Llama-3.1-8B), e.g.
#   DQ_MODEL=Qwen/Qwen3-14B DQ_LEAN_HEAD=1 ./run_all.sh
# Benchmarks run twice; report the second (warm-clock) run, as for the INT4 path.
# A failing step (e.g. OOM) is reported and the remaining steps still run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export DQ_MODEL=${DQ_MODEL:-unsloth/Meta-Llama-3.1-8B}
LOGS=logs/${DQ_MODEL##*/}
mkdir -p "$LOGS"
export HF_DATASETS_TRUST_REMOTE_CODE=1
run() { local log=$1; shift; echo "== $* -> $LOGS/$log"
        python "$@" > "$LOGS/$log" 2>&1 || echo "   FAILED (exit $?), see $LOGS/$log"; }

# correctness of the pieces (kernel test is model-independent)
run verify_kernel.log      verification/test_act_quant_bitexact.py
run verify_bcd.log         verification/test_bcd_matches_reference.py

# throughput (Table 3), per-component GPU time (Table 11), memory (Table 12)
# modes: bf16 | nvfp4 (DualQuant) | sq_nvfp4 (SmoothQuant, scale folded into RMSNorm)
#        | awq_nvfp4 (AWQ+, per-layer 1/s fused into the activation quantizer)
MODES=${MODES:-"bf16 nvfp4 sq_nvfp4 awq_nvfp4"}
case " $MODES " in *" awq_nvfp4 "*) run awq_calibration.log awq_nvfp4.py ;; esac
for i in 1 2; do
  for m in $MODES; do run graph_bench_${m}_run$i.log graph_bench_all.py --mode $m; done
done
for i in 1 2; do
  for m in $MODES; do run profile_${m}_run$i.log profile_breakdown.py --mode $m --show-kernels; done
done
for m in $MODES; do run memory_${m}.log memory_breakdown.py --mode $m; done

# accuracy (WikiText-2, seqlen 8192): fake-quant references and the real packed path
run fake_rtn_single.log    fake_quant_ppl.py --method rtn
run fake_sq_single.log     fake_quant_ppl.py --method smoothquant
run fake_awq_single.log    fake_quant_ppl.py --method awq
run fake_awq_paper.log     fake_quant_ppl.py --method awq_paper
run fake_dq_single.log     fake_quant_ppl.py --method dualquant
run real_bf16_ppl.log      ppl_eval.py --mode bf16
for m in $MODES; do [ $m = bf16 ] || run real_${m}_ppl.log ppl_eval.py --mode $m; done
grep -h "RESULT\|PREFILL\|DECODE\|ALL OK\|FAILURES" "$LOGS"/*.log
