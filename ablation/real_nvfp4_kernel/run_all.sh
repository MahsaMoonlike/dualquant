#!/bin/bash
# Regenerates every NVFP4 number (accuracy, throughput, per-component time, memory) into logs/.
# Benchmarks run twice; report the second (warm-clock) run, as for the INT4 path.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs
export HF_DATASETS_TRUST_REMOTE_CODE=1
run() { local log=$1; shift; echo "== $* -> logs/$log"; python "$@" > "logs/$log" 2>&1; }

# correctness of the pieces
run verify_kernel.log      verification/test_act_quant_bitexact.py
run verify_bcd.log         verification/test_bcd_matches_reference.py

# accuracy (WikiText-2, seqlen 8192): fake-quant references and the real packed path
run fake_bf16.log          fake_quant_ppl.py --method bf16
run fake_rtn_single.log    fake_quant_ppl.py --method rtn
run fake_dq_single.log     fake_quant_ppl.py --method dualquant
run real_bf16_ppl.log      ppl_eval.py --mode bf16
run real_nvfp4_ppl.log     ppl_eval.py --mode nvfp4

# throughput (Table 3), per-component GPU time (Table 11), memory (Table 12)
for i in 1 2; do
  run graph_bench_bf16_run$i.log   graph_bench_all.py --mode bf16
  run graph_bench_nvfp4_run$i.log  graph_bench_all.py --mode nvfp4
done
for i in 1 2; do
  run profile_bf16_run$i.log   profile_breakdown.py --mode bf16
  run profile_nvfp4_run$i.log  profile_breakdown.py --mode nvfp4 --show-kernels
done
run memory_bf16.log   memory_breakdown.py --mode bf16
run memory_nvfp4.log  memory_breakdown.py --mode nvfp4
grep -h "RESULT\|PREFILL\|DECODE\|ALL OK\|FAILURES" logs/*.log
