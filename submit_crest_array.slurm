#!/bin/bash
#SBATCH --job-name=chcl3_array
#SBATCH --partition=SingleParticleNodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch --array=1-N%K submit_crest_array.slurm <job_dirs.list> <run_crest_from_sdf.py> [crest_exe] [xtb_exe]"
  exit 1
fi

LIST_FILE="$1"
WORKFLOW_PY="$2"
CREST_EXE_ARG="${3:-}"
XTB_EXE_ARG="${4:-}"

if [[ ! -f "$LIST_FILE" ]]; then
  echo "Error: list file not found: $LIST_FILE"
  exit 1
fi
if [[ ! -f "$WORKFLOW_PY" ]]; then
  echo "Error: workflow script not found: $WORKFLOW_PY"
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
JOB_DIR="$(sed -n "${TASK_ID}p" "$LIST_FILE")"

if [[ -z "$JOB_DIR" ]] || [[ ! -d "$JOB_DIR" ]]; then
  echo "Error: invalid job directory for task $TASK_ID: $JOB_DIR"
  exit 1
fi

if [[ ! -f "$JOB_DIR/input.xyz" ]]; then
  echo "Error: missing input.xyz in $JOB_DIR"
  exit 1
fi

THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "[$(date +'%F %T')] START task=$TASK_ID dir=$JOB_DIR threads=$THREADS"
PY_ARGS=(
  --job-dir "$JOB_DIR"
  --threads "$THREADS"
)
if [[ -n "$CREST_EXE_ARG" ]]; then
  PY_ARGS+=(--crest-exe "$CREST_EXE_ARG")
fi
if [[ -n "$XTB_EXE_ARG" ]]; then
  PY_ARGS+=(--xtb-exe "$XTB_EXE_ARG")
fi

"$PYTHON_BIN" "$WORKFLOW_PY" "${PY_ARGS[@]}"
rc=$?

echo "[$(date +'%F %T')] END task=$TASK_ID dir=$JOB_DIR rc=$rc"
exit $rc
