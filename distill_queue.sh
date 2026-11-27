#!/usr/bin/env bash
# On-policy distillation queue (Nov 26-27 2026), run on the rental in tmux:
#   1. teacher and S3 answer every candidate prompt (distill_filter.py)
#   2. pick the prompts (distill_filter.py select -> data/opd/prompts.jsonl)
#   3. compare teacher prompts (train_distill.py check -> distill-check.log)
#   4. distill S3 with the 7B teacher (runs/distill)
#   5. benchmark runs/distill/final as "distill" (grade on the Mac)
# Then wait IDLE_MINUTES; unless /workspace/KEEP_RUNNING exists, stop (pause)
# this instance. Stopping keeps the disk; destroying is never done here.
set -uo pipefail
cd /workspace/pythongpt
PY=/venv/main/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IDLE_MINUTES=90
mkdir -p runs/benchmarks

log() { echo "$(date -u +%FT%TZ) $*" | tee -a queue.log; }

bench() {  # bench MODEL NAME: same protocol as S3 (greedy, 512 tokens).
  $PY benchmark_humaneval.py --dataset humaneval --model "$1" --out "runs/benchmarks/$2" > "runs/benchmarks/$2.log" 2>&1
  $PY benchmark_humaneval.py --dataset mbpp --model "$1" --out "runs/benchmarks/$2-mbpp" > "runs/benchmarks/$2-mbpp.log" 2>&1
  $PY benchmark_validation.py --model "$1" --out "runs/benchmarks/$2-val" > "runs/benchmarks/$2-val.log" 2>&1
  log "benchmarks written for $2"
}

finish() {
  log "distill queue ended; pausing in $IDLE_MINUTES min unless /workspace/KEEP_RUNNING exists"
  sleep $(( IDLE_MINUTES * 60 ))
  if [ -e /workspace/KEEP_RUNNING ]; then
    log "KEEP_RUNNING found: not stopping"
  else
    log "stopping (pausing) the instance"
    vastai stop instance "$CONTAINER_ID" 2>&1 | tee -a queue.log
  fi
}
trap finish EXIT

log "distill queue started"
$PY distill_filter.py > opd-filter.log 2>&1 || { log "prompt scoring failed"; exit 1; }
$PY distill_filter.py select > opd-select.log 2>&1 || { log "prompt selection failed"; exit 1; }
log "prompts chosen: $(grep -E '"(teacher_pass|teacher_pass_s3_fail|chosen)"' opd-select.log | tr -s ' ' | tr '\n' ' ')"
$PY train_distill.py check > distill-check.log 2>&1 || { log "distill check failed"; exit 1; }
log "distill check done: $(grep 'KL per token' distill-check.log | tr -s ' ' | tr '\n' ';')"
$PY train_distill.py > distill.log 2>&1 || { log "distill run failed"; exit 1; }
log "distill run done"
bench runs/distill/final distill
log "DISTILL QUEUE DONE"
