# CREST Array Submission Workflow

This folder runs CREST jobs in three stages:

1. Generate per-molecule job folders with `--dry-run`.
2. Submit those folders as a SLURM array.
3. Post-process CREST XYZ outputs to SDF with template-first mapping.

## Files

- `run.sh`: prepares `crest_runs/<molecule>/` folders and writes `input.xyz` + `crest_input.toml` using dry-run mode.
- `submit_crest_array.sh`: creates `job_dirs.list` and submits SLURM array jobs.
- `submit_crest_array.slurm`: executes one molecule folder per SLURM array task by calling `run_crest_from_sdf.py --job-dir`.
- `run_crest_from_sdf.py`: main workflow (xTB pre-opt + CREST run).
- `process_CREST_xyz_to_SDF_v2.py`: post-processes `crest_conformers.xyz` into SDF with template-first bond mapping, energy properties, and per-molecule run summary.

## Step 1: Prepare folders (dry-run)

Run:

```bash
bash run.sh
```

What it does:

- Reads the input SDF defined in `run.sh`.
- Creates one folder per molecule under `crest_runs/`.
- Writes `input.xyz` and `crest_input.toml`.
- Does **not** run xTB or CREST because `--dry-run` is set.

## Step 2: Submit all jobs with SLURM array

Run:

```bash
./submit_crest_array.sh \
  --outdir crest_runs \
  --max-parallel 1 \
  --crest-exe /home/pgupta11/anaconda3/envs/crest/bin/crest \
  --xtb-exe /home/pgupta11/anaconda3/envs/crest/bin/xtb
```

Required flags:

- `--outdir`: job root folder (example: `crest_runs`).
- `--max-parallel`: max concurrent array tasks.

Optional flags:

- `--sdf`: only needed if `--outdir` does not exist and you want auto-create via dry-run.
- `--crest-exe`, `--xtb-exe`: recommended on cluster nodes where binaries are not on PATH.

## Step 3: Post-process CREST outputs to SDF

After CREST jobs complete, convert XYZ conformer files to SDF format with energy properties:

```bash
python3 process_CREST_xyz_to_SDF_v2.py \
  --job-list job_dirs.list \
  --charge-sdf initial_SD_file_to_run_crest_job.sdf \
  --charge-id-prop _Name \
  --mode template-first
```

What it does:

- Reads `crest_conformers.xyz` from each job folder listed in `job_dirs.list`.
- **Template-first mode (default/recommended):**
  - Uses `--charge-sdf` as the template source.
  - Matches each job folder name to input SDF molecule ID using `--charge-id-prop` (for example `_Name`).
  - Copies template bond connectivity, bond orders, and formal charges directly, then replaces coordinates from XYZ.
  - This is more robust than pure bond inference because topology comes from the original input molecule.
- **ID match process:**
  - For each folder, script checks whether an ID match exists in the template SDF.
  - If match is found, template mapping is used.
  - If no match is found (or template validation fails), script falls back to `infer-from-xyz` for that molecule.
- **Fallback mode:**
  - You can force legacy behavior using `--mode infer-from-xyz`.
  - In this mode, bonds are inferred from XYZ + charge, with optional per-molecule charge lookup from `--charge-sdf`.
- Adds energy properties:
  - `CREST_Energy`: raw energy in Hartree
  - `CREST_Energy_kcalmol`: energy in kcal/mol
  - `CREST_RelativeEnergy_kcalmol`: relative energy (lowest - current) in kcal/mol
- Creates individual `crest_conformers.sdf` in each job folder.
- Combines all SDF files into master `crest_jobs_combined.sdf`.
- Writes a per-molecule verification CSV summary (default: `process_CREST_xyz_to_SDF_v2_run_summary.csv` beside `job_dirs.list`) with:
  - ID match status
  - template found / charge lookup found
  - whether template bond map matches XYZ-inferred bond map
  - conformers written / failed frames / notes

## TOML flags explanation

`run_crest_from_sdf.py` writes `crest_input.toml` in each job folder.

Typical TOML keys:

- `input`: geometry file read by CREST (`input.xyz`).
- `runtype`: CREST run type (default `imtd-gc`).
- `threads`: thread count used by CREST.
- `[calculation]`: calculation section container.
- `[[calculation.level]]`: method stages.
  - `method = "gfnff"` or `method = "gfn2"`.
  - `alpb = "water"` (if solvent model enabled).
  - `refine = "sp"` for the second stage in `gfn2_gfnff_sp` mode.
- `[calculation.settings]`:
  - `ewin`: energy window used for conformer filtering.
  - `rthr`: RMSD/rotamer threshold parameter.
- `[dynamics]`: dynamics block (currently no active TOML keys written by this workflow for CREST 3.0.2 compatibility).

### Important TOML compatibility notes (CREST 3.0.2)

In this workflow, these CLI options are documented but **not enforced in TOML mode**:

- `--mdlen`
- `--mrest`
- `--quick`
- `--prop reopt`

Reason: CREST 3.0.2 rejects or has version-dependent behavior for these TOML keys in this setup.

If you need strict control of those options, run in non-TOML mode:

```bash
python3 run_crest_from_sdf.py --job-dir <job_dir> --no-use-toml ...
```

## Logs and outputs

Per molecule folder (example `crest_runs/gs-XXXXXXX/`):

- `xtb.out`, `xtb.err`
- `crest.out`, `crest.err`
- `run_summary.csv`
- `crest_input.toml`

SLURM array logs (top-level folder):

- `slurm-chcl3_array-<jobid>_<taskid>.out`
- `slurm-chcl3_array-<jobid>_<taskid>.err`

## Quick troubleshooting

- Error: `CREST executable not found or not executable: crest`
  - Pass `--crest-exe` and `--xtb-exe` in `submit_crest_array.sh`.
- Error in `crest.out`: `unrecognized KEYWORD in [dynamics]-block`
  - Caused by unsupported TOML keys for your CREST version.
  - Use current script version (which avoids unsupported keys) or use `--no-use-toml`.
- Missing `input.xyz`
  - Re-run `bash run.sh` to regenerate folders.
