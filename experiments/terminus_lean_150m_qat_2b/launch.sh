#!/usr/bin/env bash
# Resumable, fresh-token, ternary-weight QAT continuation of the verified 10B run.
set -Eeuo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the persistent packed-shard directory}
PARENT_ROOT=${PARENT_ROOT:?set PARENT_ROOT to the verified dense run directory}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to the persistent QAT run directory}
S3_RUN_URI=${S3_RUN_URI:?set S3_RUN_URI to the QAT artifact mirror prefix}
HERE="$REPO/experiments/terminus_lean_150m_qat_2b"
CONFIG="$HERE/configs/lean_ternary_qat.json"
mkdir -p "$RUN_ROOT"

aws sts get-caller-identity >/dev/null
"$PYTHON" "$HERE/plan.py" check
test -f "$DATA_ROOT/manifest.json"
test -f "$PARENT_ROOT/audit_final.json"
test -f "$PARENT_ROOT/dense/checkpoints/step_00025432.pt"

mirror_once() {
  aws s3 sync "$RUN_ROOT" "$S3_RUN_URI" --exclude '.*' --exclude '*.tmp' \
    --exclude 'mirror.log' --only-show-errors >> "$RUN_ROOT/mirror.log" 2>&1
}
mirror_loop() {
  while true; do
    mirror_once || printf '%s mirror failed; local artifacts remain\n' "$(date -u +%FT%TZ)" >> "$RUN_ROOT/mirror.log"
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
def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
Path(sys.argv[1]).write_text(json.dumps({
    "time_unix": time.time(), "git_commit": command("git", "rev-parse", "HEAD"),
    "gpu": command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
    "cuda_compiler": command(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2") + "/bin/nvcc", "--version"),
}, indent=2) + "\n")
PY
mirror_loop &
MIRROR_PID=$!

if [ ! -f "$RUN_ROOT/views/manifest.qat.json" ]; then
  "$PYTHON" "$HERE/prepare_views.py" --base-manifest "$DATA_ROOT/manifest.json" \
    --dense-manifest "$PARENT_ROOT/views/manifest.dense.json" \
    --parent-run "$PARENT_ROOT/dense" --data-root "$DATA_ROOT" \
    --output-dir "$RUN_ROOT/views" > "$RUN_ROOT/data_gate.log" 2>&1
fi
MANIFEST="$RUN_ROOT/views/manifest.qat.json"

if [ ! -f "$RUN_ROOT/preflight.json" ]; then
  "$PYTHON" -m terminus.preflight --config "$CONFIG" \
    --output "$RUN_ROOT/preflight.json" > "$RUN_ROOT/preflight.log" 2>&1
fi

if [ ! -f "$RUN_ROOT/qat/checkpoints/step_00000000.json" ]; then
  "$PYTHON" -m terminus.transition --parent-run "$PARENT_ROOT/dense" \
    --parent-checkpoint "$PARENT_ROOT/dense/checkpoints/step_00025432.pt" \
    --config "$CONFIG" --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
    --output-dir "$RUN_ROOT/qat" --device cuda --data-policy restart \
    > "$RUN_ROOT/transition.log" 2>&1
fi

if [ ! -f "$RUN_ROOT/qat_initial_eval.json" ]; then
  "$PYTHON" "$HERE/initial_eval.py" --run-dir "$RUN_ROOT/qat" \
    --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
    --output "$RUN_ROOT/qat_initial_eval.json" > "$RUN_ROOT/qat_initial_eval.log" 2>&1
fi

if [ ! -f "$RUN_ROOT/qat/checkpoints/step_00005087.json" ]; then
  "$PYTHON" "$REPO/experiments/terminus_lean_150m_10b/resume_prepare.py" \
    --run-dir "$RUN_ROOT/qat" >> "$RUN_ROOT/resume.log" 2>&1
  "$PYTHON" -m terminus.train --config "$RUN_ROOT/qat/qat_config.json" \
    --manifest "$MANIFEST" --data-root "$DATA_ROOT" --output-dir "$RUN_ROOT/qat" \
    --preflight-report "$RUN_ROOT/preflight.json" --resume "$RUN_ROOT/qat/checkpoints" \
    >> "$RUN_ROOT/training.log" 2>&1
fi

"$PYTHON" "$HERE/audit.py" --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/audit_final.json" > "$RUN_ROOT/audit_final.log" 2>&1
