#!/usr/bin/env bash
# One resumable dense-FP8 run. Invoke only after the 10B packed set is complete.
set -Eeuo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the persistent packed-shard directory}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to the persistent run directory}
S3_RUN_URI=${S3_RUN_URI:?set S3_RUN_URI to the run-artifact mirror prefix}
HERE="$REPO/experiments/terminus_lean_150m_10b"
CONFIG="$HERE/configs/lean_dense.json"
MIDPOINT=12716
FINAL=25432
mkdir -p "$RUN_ROOT"

aws sts get-caller-identity >/dev/null
"$PYTHON" "$HERE/plan.py" check --microbatch 48
test -f "$DATA_ROOT/manifest.json"

mirror_once() {
  aws s3 sync "$RUN_ROOT" "$S3_RUN_URI" --exclude '.*' --exclude '*.tmp' \
    --exclude 'mirror.log' --only-show-errors >> "$RUN_ROOT/mirror.log" 2>&1
}
mirror_loop() {
  while true; do
    mirror_once || printf '%s mirror failed; durable local artifacts remain\n' "$(date -u +%FT%TZ)" >> "$RUN_ROOT/mirror.log"
    sleep 300
  done
}
MIRROR_PID=
finish() {
  result=$?
  trap - EXIT
  if [ -n "$MIRROR_PID" ]; then
    kill "$MIRROR_PID" 2>/dev/null || true
    wait "$MIRROR_PID" 2>/dev/null || true
  fi
  mirror_once || { if [ "$result" -eq 0 ]; then result=1; fi; }
  exit "$result"
}
trap finish EXIT

cp "$CONFIG" "$RUN_ROOT/config.json"
cp "$HERE/plan.json" "$RUN_ROOT/plan.json"
"$PYTHON" - "$RUN_ROOT/environment.json" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path
out = Path(sys.argv[1])
def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
out.write_text(json.dumps({
    "time_unix": time.time(),
    "git_commit": command("git", "rev-parse", "HEAD"),
    "gpu": command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
    "cuda_compiler": command(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2") + "/bin/nvcc", "--version"),
}, indent=2) + "\n")
PY
mirror_loop &
MIRROR_PID=$!

if [ ! -f "$RUN_ROOT/views/manifest.dense.json" ]; then
  "$PYTHON" "$HERE/prepare_views.py" --base-manifest "$DATA_ROOT/manifest.json" \
    --data-root "$DATA_ROOT" --output-dir "$RUN_ROOT/views" \
    > "$RUN_ROOT/data_gate.log" 2>&1
fi
MANIFEST="$RUN_ROOT/views/manifest.dense.json"
if [ ! -f "$RUN_ROOT/preflight.json" ]; then
  "$PYTHON" -m terminus.preflight --config "$CONFIG" \
    --output "$RUN_ROOT/preflight.json" > "$RUN_ROOT/preflight.log" 2>&1
fi

run_train() {
  local log=$1
  shift
  local resume=()
  if [ -f "$RUN_ROOT/dense/run.json" ]; then
    if [ -f "$RUN_ROOT/dense/checkpoints/latest.json" ]; then
      "$PYTHON" "$HERE/resume_prepare.py" --run-dir "$RUN_ROOT/dense"
      resume=(--resume "$RUN_ROOT/dense/checkpoints")
    else
      # Preserve all records if interrupted before the first committed checkpoint.
      mv "$RUN_ROOT/dense" "$RUN_ROOT/dense_before_first_checkpoint_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
  fi
  "$PYTHON" -m terminus.train --config "$CONFIG" --manifest "$MANIFEST" \
    --data-root "$DATA_ROOT" --output-dir "$RUN_ROOT/dense" \
    --preflight-report "$RUN_ROOT/preflight.json" "${resume[@]}" "$@" \
    >> "$RUN_ROOT/$log" 2>&1
}

if [ ! -f "$RUN_ROOT/dense/checkpoints/step_00012716.json" ]; then
  run_train phase1.log --stop-after-updates "$MIDPOINT"
fi
"$PYTHON" "$HERE/audit.py" --run-dir "$RUN_ROOT/dense" --phase midpoint \
  --output "$RUN_ROOT/audit_midpoint.json" > "$RUN_ROOT/audit_midpoint.log" 2>&1
mirror_once

if [ ! -f "$RUN_ROOT/dense/checkpoints/step_00025432.json" ]; then
  run_train phase2.log
fi
"$PYTHON" "$HERE/audit.py" --run-dir "$RUN_ROOT/dense" --phase final \
  --output "$RUN_ROOT/audit_final.json" > "$RUN_ROOT/audit_final.log" 2>&1
