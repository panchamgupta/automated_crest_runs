#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

JOBS_ROOT=""
MAX_PARALLEL=""
SDF_FILE=""
CREST_EXE=""
XTB_EXE=""

usage() {
  cat <<'EOF'
Usage:
  ./submit_crest_array.sh --outdir DIR --max-parallel N [--sdf FILE] [--crest-exe PATH] [--xtb-exe PATH]

Examples:
  ./submit_crest_array.sh --outdir crest_jobs --max-parallel 5
  ./submit_crest_array.sh --outdir crest_jobs_v2 --sdf indane_macrocycle_list.sdf --max-parallel 6
  ./submit_crest_array.sh --outdir crest_runs --max-parallel 1 --crest-exe /home/pgupta11/anaconda3/envs/crest/bin/crest --xtb-exe /home/pgupta11/anaconda3/envs/crest/bin/xtb

Behavior:
  - If --outdir exists, submit array from existing molecule folders.
  - If --outdir does not exist, create molecule folders first using Python dry-run,
    then submit the array.

Notes:
  - Required arguments: --outdir, --max-parallel
  - --sdf is required only when --outdir does not already exist.
  - Use --crest-exe/--xtb-exe when compute nodes do not have crest/xtb on PATH.
  - Chemistry/runtime parameters are defined only in run_crest_from_sdf.py.
  - This script only prepares folders and submits the SLURM array.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)
      [[ $# -ge 2 ]] || { echo "Error: --outdir needs a value"; exit 1; }
      JOBS_ROOT="$2"
      shift 2
      ;;
    --max-parallel)
      [[ $# -ge 2 ]] || { echo "Error: --max-parallel needs a value"; exit 1; }
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --sdf)
      [[ $# -ge 2 ]] || { echo "Error: --sdf needs a value"; exit 1; }
      SDF_FILE="$2"
      shift 2
      ;;
    --crest-exe)
      [[ $# -ge 2 ]] || { echo "Error: --crest-exe needs a value"; exit 1; }
      CREST_EXE="$2"
      shift 2
      ;;
    --xtb-exe)
      [[ $# -ge 2 ]] || { echo "Error: --xtb-exe needs a value"; exit 1; }
      XTB_EXE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$JOBS_ROOT" ]]; then
  echo "Error: --outdir is required"
  usage
  exit 1
fi

if [[ -z "$MAX_PARALLEL" ]]; then
  echo "Error: --max-parallel is required"
  usage
  exit 1
fi

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "Error: max parallel must be a positive integer (got: $MAX_PARALLEL)"
  exit 1
fi

if [[ ! -d "$JOBS_ROOT" ]]; then
  if [[ -z "$SDF_FILE" ]]; then
    echo "Error: jobs root not found: $JOBS_ROOT"
    echo "Error: --sdf is required to auto-create jobs when outdir does not exist"
    exit 1
  fi

  if [[ ! -f "$SDF_FILE" ]]; then
    echo "Error: jobs root not found: $JOBS_ROOT"
    echo "Error: SDF file for auto-create not found: $SDF_FILE"
    echo "Tip: pass --sdf <path_to_sdf> or create $JOBS_ROOT first."
    exit 1
  fi

  echo "[INFO] Jobs root not found: $JOBS_ROOT"
  echo "[INFO] Creating molecule folders from SDF via dry-run..."
  python3 run_crest_from_sdf.py \
    --sdf "$SDF_FILE" \
    --outdir "$JOBS_ROOT" \
    --dry-run
fi

LIST_FILE="${JOBS_ROOT%/}/job_dirs.list"

if [[ ! -d "$JOBS_ROOT" ]]; then
  echo "Error: jobs root not found: $JOBS_ROOT"
  echo "Tip: pass an existing folder, e.g. ./submit_crest_array.sh --outdir crest_jobs_toml_check --max-parallel 5"
  exit 1
fi

# Build task list from existing molecule folders that already contain input.xyz.
find "$JOBS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | while read -r d; do
  if [[ -f "$d/input.xyz" ]]; then
    echo "$d"
  fi
done > "$LIST_FILE"

TOTAL_TASKS="$(wc -l < "$LIST_FILE" | tr -d ' ')"
if [[ "$TOTAL_TASKS" -eq 0 ]]; then
  echo "Error: no molecule folders with input.xyz found under $JOBS_ROOT"
  exit 1
fi

echo "Submitting array: tasks=$TOTAL_TASKS, max_concurrent=$MAX_PARALLEL"
echo "Task list: $LIST_FILE"
echo "Jobs root: $JOBS_ROOT"

WORKFLOW_PY="$SCRIPT_DIR/run_crest_from_sdf.py"
if [[ ! -f "$WORKFLOW_PY" ]]; then
  echo "Error: workflow script not found: $WORKFLOW_PY"
  exit 1
fi

SBATCH_ARGS=(
  --array="1-${TOTAL_TASKS}%${MAX_PARALLEL}"
  "$SCRIPT_DIR/submit_crest_array.slurm"
  "$LIST_FILE"
  "$WORKFLOW_PY"
)

if [[ -n "$CREST_EXE" ]]; then
  SBATCH_ARGS+=("$CREST_EXE")
fi
if [[ -n "$XTB_EXE" ]]; then
  if [[ -z "$CREST_EXE" ]]; then
    SBATCH_ARGS+=("")
  fi
  SBATCH_ARGS+=("$XTB_EXE")
fi

sbatch "${SBATCH_ARGS[@]}"
