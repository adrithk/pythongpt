#!/usr/bin/env bash
# GRPO experiment queue (Nov 26 2026), run on the rental in tmux:
#   1. wait for the 1e-6 run (runs/grpo), benchmark it
#   2. score every pool task with S3 (grpo_filter.py -> data/grpo/pool-mixed.jsonl)
#   3. 3e-6 run from S3 on the same pool (runs/grpo-lr3e-6), benchmark it
#   4. run on the mixed pool at the learning rate whose final held-out score
#      was higher (runs/grpo-mixed), benchmark it
# Then wait IDLE_MINUTES; unless /workspace/KEEP_RUNNING exists, stop (pause)
# this instance. Stopping keeps the disk; destroying is never done here.
set -uo pipefail
cd /workspace/pythongpt
PY=/venv/main/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IDLE_MINUTES=90
mkdir -p runs/benchmarks

log() { echo "$(date -u +%FT%TZ) $*" | tee -a queue.log; }

# variant OUT KEY=VALUE...: a copy of train_grpo.py with settings replaced.
variant() {
  local out=$1; shift
  cp train_grpo.py "$out"
  for kv in "$@"; do
    local k=${kv%%=*} v=${kv#*=}
    sed -i "s|^$k = .*|$k = $v|" "$out"
    grep -qF "$k = $v" "$out" || { log "could not set $k in $out"; exit 1; }
  done
}

# bench MODEL NAME: same protocol as S3 (greedy, 512 tokens); grade on the Mac.
bench() {
  $PY benchmark_humaneval.py --dataset humaneval --model "$1" --out "runs/benchmarks/$2" > "runs/benchmarks/$2.log" 2>&1
  $PY benchmark_humaneval.py --dataset mbpp --model "$1" --out "runs/benchmarks/$2-mbpp" > "runs/benchmarks/$2-mbpp.log" 2>&1
  $PY benchmark_validation.py --model "$1" --out "runs/benchmarks/$2-val" > "runs/benchmarks/$2-val.log" 2>&1
  log "benchmarks written for $2"
}

final_dev() {  # The last held-out pass rate in a run's log.jsonl.
  $PY -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1])]; print([x['dev_pass_rate'] for x in r if 'dev_pass_rate' in x][-1])" "$1/log.jsonl"
}

# On any exit (done or failed): wait, then pause the GPU unless told not to.
finish() {
  log "queue ended; pausing in $IDLE_MINUTES min unless /workspace/KEEP_RUNNING exists"
  sleep $(( IDLE_MINUTES * 60 ))
  if [ -e /workspace/KEEP_RUNNING ]; then
    log "KEEP_RUNNING found: not stopping"
  else
    log "stopping (pausing) the instance"
    vastai stop instance "$CONTAINER_ID" 2>&1 | tee -a queue.log
  fi
}
trap finish EXIT

log "queue started"
while tmux has-session -t grpo 2>/dev/null; do sleep 30; done
[ -d runs/grpo/final ] || { log "runs/grpo/final missing: 1e-6 run failed"; exit 1; }
log "1e-6 run done, final held-out $(final_dev runs/grpo)"
bench runs/grpo/final grpo-lr1e-6 &

log "filter started"
$PY grpo_filter.py >> filter.log 2>&1 || { log "filter failed"; exit 1; }
log "filter done: $(tail -3 filter.log | tr '\n' ' ')"

variant train_grpo_lr3.py 'LEARNING_RATE=3e-6' 'OUTPUT=Path("runs/grpo-lr3e-6")' 'MAX_TOKENS=8_000'
log "3e-6 run started"
$PY train_grpo_lr3.py > grpo-lr3e-6.log 2>&1 || { log "3e-6 run failed"; exit 1; }
log "3e-6 run done, final held-out $(final_dev runs/grpo-lr3e-6)"
bench runs/grpo-lr3e-6/final grpo-lr3e-6 &

LR=$($PY -c "import sys; print('3e-6' if float(sys.argv[2]) > float(sys.argv[1]) else '1e-6')" "$(final_dev runs/grpo)" "$(final_dev runs/grpo-lr3e-6)")
MIXED=$(wc -l < data/grpo/pool-mixed.jsonl)
STEPS=$(( MIXED * 2 / 16 )); [ "$STEPS" -gt 300 ] && STEPS=300  # About 2 passes, at most 300.
variant train_grpo_mixed.py "LEARNING_RATE=$LR" 'POOL="data/grpo/pool-mixed.jsonl"' 'OUTPUT=Path("runs/grpo-mixed")' 'MAX_TOKENS=8_000' "STEPS=$STEPS"
log "mixed-pool run started: $MIXED tasks, lr $LR, $STEPS steps"
$PY train_grpo_mixed.py > grpo-mixed.log 2>&1 || { log "mixed-pool run failed"; exit 1; }
log "mixed-pool run done, final held-out $(final_dev runs/grpo-mixed)"
bench runs/grpo-mixed/final grpo-mixed

wait
log "QUEUE DONE"
