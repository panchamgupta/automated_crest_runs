#!/usr/bin/env python3
"""Batch SDF -> XYZ -> CREST workflow with isolated per-molecule outputs.

Key behavior:
- Reads all molecules from one SDF.
- Creates a unique directory per molecule to avoid output collisions.
- Writes one XYZ input per directory.
- Runs CREST inside each directory so default filenames never overwrite each other.
- Writes a run summary CSV with status and return codes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem


DEFAULT_OUTDIR = "crest_jobs"


def sanitize_name(text: str, max_len: int = 48) -> str:
    """Return filesystem-safe lowercase token."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    safe = safe.strip("._-")
    if not safe:
        safe = "mol"
    return safe[:max_len].lower()


def molecule_label(mol: Chem.Mol, idx1: int, name_prop: str) -> str:
    """Choose display label from selected property with robust fallbacks."""
    for prop in (name_prop, "_Name", "Title", "name"):
        if mol.HasProp(prop):
            value = str(mol.GetProp(prop)).strip()
            if value:
                return value
    return f"mol_{idx1:04d}"


def ensure_3d_coordinates(mol: Chem.Mol, embed_missing: bool) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """Ensure molecule has one 3D conformer suitable for XYZ writing."""
    if mol is None:
        return None, "invalid molecule record"

    work = Chem.Mol(mol)

    if work.GetNumConformers() == 0:
        if not embed_missing:
            return None, "missing conformer and --embed-missing not enabled"
        work = Chem.AddHs(work)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        status = AllChem.EmbedMolecule(work, params)
        if status != 0:
            return None, "ETKDG embedding failed"
        try:
            AllChem.UFFOptimizeMolecule(work, maxIters=200)
        except Exception:
            pass

    conf = work.GetConformer()
    if not conf.Is3D() and embed_missing:
        work = Chem.AddHs(work, addCoords=True)
        try:
            AllChem.UFFOptimizeMolecule(work, maxIters=200)
        except Exception:
            pass

    if work.GetNumConformers() == 0:
        return None, "no conformer available after processing"

    return work, None


def unique_job_dir(base_out: Path, idx1: int, label: str) -> Path:
    """Build label-based directory name; add numeric suffix only on collision."""
    token = sanitize_name(label)
    if not token:
        token = f"mol_{idx1:04d}"

    candidate = base_out / token
    if not candidate.exists():
        return candidate

    serial = 2
    while True:
        candidate = base_out / f"{token}_{serial}"
        if not candidate.exists():
            return candidate
        serial += 1


def resolve_executable(cli_value: Optional[str], env_key: str, default_name: str) -> str:
    """Resolve executable from CLI, environment variable, or PATH lookup."""
    if cli_value:
        return cli_value
    env_value = os.environ.get(env_key)
    if env_value:
        return env_value
    which_path = shutil.which(default_name)
    return which_path if which_path else default_name


def validate_executable(exe_value: str, exe_label: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve and validate executable path."""
    resolved = shutil.which(exe_value)
    if resolved is not None:
        return resolved, None

    exe_path = Path(exe_value).expanduser()
    if exe_path.is_file() and os.access(str(exe_path), os.X_OK):
        return str(exe_path.resolve()), None

    return None, f"{exe_label} executable not found or not executable: {exe_value}"


def get_crest_version(crest_exe: str) -> Optional[str]:
    """Try to get CREST version string from executable."""
    for cmd in ([crest_exe, "--version"], [crest_exe, "-V"]):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        except Exception:
            continue
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        for line in output.splitlines():
            line = line.strip()
            if line:
                return line
    return None


def normalize_alpb(alpb: Optional[str]) -> Optional[str]:
    """Normalize ALPB solvent selection; return None for vacuum/none."""
    if alpb is None:
        return None
    value = str(alpb).strip()
    if value.lower() in {"", "none", "vacuum"}:
        return None
    return value


def write_crest_toml(
    toml_path,
    input_xyz="input.xyz",
    threads=32,
    alpb="water",
    ewin=10.0,
    rthr=0.20,
    mdlen="x2.0",
    mrest=3,
    search_method="gfn2_gfnff_sp",
    runtype="imtd-gc",
    quick_requested=False,
    prop="reopt",
) -> str:
    """Write CREST 3 TOML input and return written content."""
    alpb_value = normalize_alpb(alpb)
    lines = [
        f'input = "{input_xyz}"',
        f'runtype = "{runtype}"',
        f"threads = {int(threads)}",
        "",
        "[calculation]",
        "",
    ]

    def add_level(method: str, refine: Optional[str] = None) -> None:
        lines.append("[[calculation.level]]")
        lines.append(f'method = "{method}"')
        if alpb_value is not None:
            lines.append(f'alpb = "{alpb_value}"')
        if refine is not None:
            lines.append(f'refine = "{refine}"')
        lines.append("")

    if search_method == "gfnff":
        add_level("gfnff")
    elif search_method == "gfn2":
        add_level("gfn2")
    elif search_method == "gfn2_gfnff_sp":
        add_level("gfnff")
        add_level("gfn2", refine="sp")
    else:
        raise ValueError(f"Unsupported search_method: {search_method}")

    lines.extend(
        [
            "[calculation.settings]",
            f"ewin = {float(ewin)}",
            f"rthr = {float(rthr)}",
        ]
    )

    lines.extend(
        [
            "",
            "[dynamics]",
        ]
    )

    lines.extend(
        [
            "",
            "# mdlen was requested but is not written in TOML mode because",
            "# CREST 3.0.2 rejects mdlen in the [dynamics] block.",
            "# Use --no-use-toml fallback mode if strict mdlen control is required.",
        ]
    )

    lines.extend(
        [
            "",
            "# mrest was requested but is not written in TOML mode because",
            "# CREST 3.0.2 rejects mrest in the [dynamics] block.",
            "# Use --no-use-toml fallback mode if strict mrest control is required.",
        ]
    )

    if quick_requested:
        lines.extend(
            [
                "",
                "# --quick was requested, but quick-mode TOML key is version-dependent in CREST 3.",
                "# Quick mode is applied only in non-TOML fallback mode (--no-use-toml).",
            ]
        )

    if str(prop).strip().lower() == "reopt":
        lines.extend(
            [
                "",
                "# --prop reopt requested.",
                "# TOML key for reoptimization is not applied here to avoid unsupported keywords.",
                "# Post-reoptimization step is skipped in TOML mode.",
            ]
        )

    content = "\n".join(lines) + "\n"
    Path(toml_path).write_text(content, encoding="utf-8")
    return content


def build_crest_command(crest_exe, use_toml=True):
    """Build base CREST command for TOML or legacy CLI mode."""
    return [crest_exe, "--input", "crest_input.toml"] if use_toml else [crest_exe]


def make_scratch_dir(scratch_root: Optional[Path], mol_dir_name: str) -> Optional[Path]:
    """Return per-molecule scratch directory path if scratch root is configured."""
    if scratch_root is None:
        return None
    d = scratch_root / mol_dir_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_from_existing_dir(job_dir: Path) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """Build a job dict from an existing job directory with input.xyz."""
    if not job_dir.is_dir():
        return None, f"job directory not found: {job_dir}"

    input_xyz = job_dir / "input.xyz"
    if not input_xyz.is_file() or input_xyz.stat().st_size == 0:
        return None, f"missing input.xyz in job directory: {job_dir}"

    return {
        "index": 1,
        "label": job_dir.name,
        "job_dir": str(job_dir),
    }, None


def run_crest_for_molecule(
    mol_dir: Path,
    xyz_name: str,
    label: str,
    crest_exe: str,
    xtb_exe: str,
    threads: int,
    alpb: str,
    quick: bool,
    ewin: float,
    rthr: float,
    mdlen: str,
    mrest: int,
    prop: str,
    use_toml: bool,
    search_method: str,
    runtype: str,
    scratch_dir: Optional[Path],
    dry_run: bool,
) -> Tuple[int, str, str, Optional[Path], Optional[str]]:
    """Run CREST in molecule directory and return (code, message)."""
    input_xyz_path = mol_dir / xyz_name
    if not input_xyz_path.exists() or input_xyz_path.stat().st_size == 0:
        return 4, f"missing CREST input XYZ: {xyz_name}", "", None, None

    toml_path: Optional[Path] = None
    toml_content: Optional[str] = None
    if use_toml:
        toml_path = mol_dir / "crest_input.toml"
        toml_content = write_crest_toml(
            toml_path=toml_path,
            input_xyz=xyz_name,
            threads=threads,
            alpb=alpb,
            ewin=ewin,
            rthr=rthr,
            mdlen=mdlen,
            mrest=mrest,
            search_method=search_method,
            runtype=runtype,
            quick_requested=quick,
            prop=prop,
        )
        cmd = build_crest_command(crest_exe, use_toml=True)
    else:
        method_flag = {
            "gfnff": "--gfnff",
            "gfn2": "--gfn2",
            "gfn2_gfnff_sp": "--gfn2//gfnff",
        }[search_method]
        cmd = build_crest_command(crest_exe, use_toml=False)
        cmd.extend(
            [
                xyz_name,
                method_flag,
                "--prop",
                prop,
                "--ewin",
                str(ewin),
                "--rthr",
                str(rthr),
                "--mdlen",
                str(mdlen),
                "--mrest",
                str(mrest),
                "--T",
                str(threads),
            ]
        )
        alpb_value = normalize_alpb(alpb)
        if alpb_value is not None:
            cmd.extend(["--alpb", alpb_value])
        if quick:
            cmd.append("--quick")

    if scratch_dir is not None:
        cmd.extend(["--scratch", str(scratch_dir)])

    cmd_text = shlex.join(cmd)

    if dry_run:
        print(f"[DRY-RUN] Molecule: {label}")
        print(f"[DRY-RUN] Run directory: {mol_dir}")
        if toml_path is not None and toml_content is not None:
            print(f"[DRY-RUN] TOML path: {toml_path}")
            print("[DRY-RUN] TOML contents:")
            print(toml_content.rstrip())
        print(f"[DRY-RUN] Command: {cmd_text}")
        return 0, "dry-run", cmd_text, toml_path, toml_content

    env = os.environ.copy()
    # CREST typically calls xtb from PATH; prepend xtb and crest locations for reliability.
    extra_dirs = [str(Path(xtb_exe).parent), str(Path(crest_exe).parent)]
    env["PATH"] = os.pathsep.join(extra_dirs + [env.get("PATH", "")])
    env["OMP_NUM_THREADS"] = str(threads)
    # Avoid nested BLAS thread storms and OpenBLAS warning spam on stderr.
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["GOTO_NUM_THREADS"] = "1"

    with (mol_dir / "crest.out").open("w", encoding="utf-8") as fout, (mol_dir / "crest.err").open("w", encoding="utf-8") as ferr:
        proc = subprocess.run(cmd, cwd=mol_dir, stdout=fout, stderr=ferr, env=env)
    if proc.returncode == 0:
        return 0, "ok", cmd_text, toml_path, toml_content
    return proc.returncode, "crest failed (see crest.err)", cmd_text, toml_path, toml_content


def run_xtb_preopt_for_molecule(
    mol_dir: Path,
    input_xyz_name: str,
    output_xyz_name: str,
    xtb_exe: str,
    threads: int,
    alpb: str,
    dry_run: bool,
) -> Tuple[int, str]:
    """Run xTB pre-optimization and create the CREST input filename."""
    cmd = [xtb_exe, input_xyz_name, "--opt", "--gfnff"]
    alpb_value = normalize_alpb(alpb)
    if alpb_value is not None:
        cmd.extend(["--alpb", alpb_value])

    if dry_run:
        return 0, "dry-run"

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(Path(xtb_exe).parent), env.get("PATH", "")])
    env["OMP_NUM_THREADS"] = str(threads)
    # Keep OpenMP workers for xTB, but force BLAS libraries to single-thread.
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["GOTO_NUM_THREADS"] = "1"

    with (mol_dir / "xtb.out").open("w", encoding="utf-8") as fout, (mol_dir / "xtb.err").open("w", encoding="utf-8") as ferr:
        proc = subprocess.run(cmd, cwd=mol_dir, stdout=fout, stderr=ferr, env=env)

    if proc.returncode != 0:
        return proc.returncode, "xtb pre-opt failed (see xtb.err)"

    # xTB output filename can vary by version/workflow.
    # Prefer inputopt.xyz if present, otherwise fall back to xtbopt.xyz.
    optimized_candidates = [mol_dir / "inputopt.xyz", mol_dir / "xtbopt.xyz"]
    optimized_xyz = None
    for candidate in optimized_candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            optimized_xyz = candidate
            break
    if optimized_xyz is None:
        return 3, "xtb finished but neither inputopt.xyz nor xtbopt.xyz was found"

    shutil.copy2(optimized_xyz, mol_dir / output_xyz_name)
    return 0, "ok"


def run_pipeline_for_molecule(
    job: Dict[str, object],
    crest_exe: str,
    xtb_exe: str,
    threads: int,
    alpb: str,
    quick: bool,
    ewin: float,
    rthr: float,
    mdlen: str,
    mrest: int,
    prop: str,
    use_toml: bool,
    search_method: str,
    runtype: str,
    scratch_root: Optional[Path],
    dry_run: bool,
) -> Dict[str, object]:
    """Run xTB pre-opt then CREST for a prepared molecule job."""
    idx1 = int(job["index"])
    label = str(job["label"])
    job_dir = Path(str(job["job_dir"]))

    input_xyz_name = "input.xyz"
    crest_input_name = "macrocycle_preopt.xyz"

    xtb_code, xtb_msg = run_xtb_preopt_for_molecule(
        mol_dir=job_dir,
        input_xyz_name=input_xyz_name,
        output_xyz_name=crest_input_name,
        xtb_exe=xtb_exe,
        threads=threads,
        alpb=alpb,
        dry_run=dry_run,
    )

    if xtb_code != 0:
        print(f"[END] {label} return_code={xtb_code}")
        return {
            "index": idx1,
            "label": label,
            "job_dir": str(job_dir),
            "status": "fail",
            "xtb_return_code": xtb_code,
            "crest_return_code": "",
            "message": xtb_msg,
        }

    # In TOML mode, keep template input path as local input.xyz while still
    # using the xTB-optimized geometry produced above.
    crest_xyz_name = crest_input_name
    if use_toml:
        optimized_path = job_dir / crest_input_name
        input_path = job_dir / input_xyz_name
        if optimized_path.exists() and optimized_path.stat().st_size > 0:
            shutil.copy2(optimized_path, input_path)
        crest_xyz_name = input_xyz_name

    scratch_dir = make_scratch_dir(scratch_root, job_dir.name)
    print(f"[START] {label}")
    crest_code, crest_msg, crest_cmd, _, _ = run_crest_for_molecule(
        mol_dir=job_dir,
        xyz_name=crest_xyz_name,
        label=label,
        crest_exe=crest_exe,
        xtb_exe=xtb_exe,
        threads=threads,
        alpb=alpb,
        quick=quick,
        ewin=ewin,
        rthr=rthr,
        mdlen=mdlen,
        mrest=mrest,
        prop=prop,
        use_toml=use_toml,
        search_method=search_method,
        runtype=runtype,
        scratch_dir=scratch_dir,
        dry_run=dry_run,
    )
    print(f"[COMMAND] {crest_cmd}")
    print(f"[END] {label} return_code={crest_code}")

    return {
        "index": idx1,
        "label": label,
        "job_dir": str(job_dir),
        "status": "ok" if crest_code == 0 else "fail",
        "xtb_return_code": xtb_code,
        "crest_return_code": crest_code,
        "message": crest_msg,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run xTB pre-opt and CREST with isolated per-molecule outputs. "
            "Use either --sdf (batch mode) or --job-dir (single existing folder mode)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Example 1: balanced macrocycle search in implicit water\n"
            "    python run_crest_from_sdf.py --sdf input.sdf --outdir crest_runs "
            "--crest-exe /home/pgupta11/anaconda3/envs/crest/bin/crest "
            "--use-toml --search-method gfn2_gfnff_sp --alpb water --ewin 6.0 "
            "--rthr 0.20 --mrest 3 --threads 32\n\n"
            "  Example 2: fast GFN-FF only\n"
            "    python run_crest_from_sdf.py --sdf input.sdf --outdir crest_runs "
            "--crest-exe /home/pgupta11/anaconda3/envs/crest/bin/crest "
            "--use-toml --search-method gfnff --alpb water --ewin 5.0 "
            "--rthr 0.25 --mrest 2 --threads 32\n\n"
            "Note: TOML keys for --quick and --prop reopt are version-dependent in CREST 3. "
            "This script documents these in crest_input.toml and applies them only in "
            "--no-use-toml fallback mode."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--sdf", help="Input SDF file containing one or more molecules.")
    mode_group.add_argument(
        "--job-dir",
        help="Run only one existing job directory that already contains input.xyz.",
    )

    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help=f"Base output directory (default: {DEFAULT_OUTDIR}).")
    parser.add_argument("--name-prop", default="_Name", help="Property used for molecule naming (default: _Name).")
    parser.add_argument("--max-mols", type=int, default=None, help="Optional cap on molecules to process.")
    parser.add_argument("--embed-missing", action="store_true", help="Embed 3D coordinates when missing.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare folders and XYZ files, skip CREST execution.")

    parser.add_argument(
        "--crest-exe",
        default=None,
        help="Path to crest executable (default: CREST_EXE env var or PATH lookup).",
    )
    parser.add_argument(
        "--xtb-exe",
        default=None,
        help="Path to xtb executable (default: XTB_EXE env var or PATH lookup).",
    )
    parser.add_argument("--legacy", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--threads", type=int, default=16, help="Threads passed to CREST via --T (default: 16).")
    parser.set_defaults(use_toml=True)
    parser.add_argument(
        "--use-toml",
        dest="use_toml",
        action="store_true",
        help="Use CREST 3 TOML input mode (default: enabled).",
    )
    parser.add_argument(
        "--no-use-toml",
        dest="use_toml",
        action="store_false",
        help="Disable TOML mode and use command-line CREST flags (fallback mode).",
    )
    parser.add_argument(
        "--search-method",
        choices=["gfnff", "gfn2", "gfn2_gfnff_sp"],
        default="gfn2_gfnff_sp",
        help="Search method profile for TOML or CLI fallback (default: gfn2_gfnff_sp).",
    )
    parser.add_argument(
        "--runtype",
        default="imtd-gc",
        help="CREST runtype used in TOML mode (default: imtd-gc).",
    )
    parser.add_argument(
        "--max-parallel-molecules",
        type=int,
        default=1,
        help="Number of molecules to run concurrently (default: 1).",
    )
    parser.add_argument("--alpb", default="water", help="ALPB solvent (default: water).")
    parser.add_argument("--ewin", type=float, default=5.0, help="CREST --ewin value (default: 5.0).")
    parser.add_argument("--rthr", type=float, default=0.20, help="CREST --rthr value (default: 0.20).")
    parser.add_argument("--mdlen", default="x2.0", help="CREST --mdlen value (default: x2.0).")
    parser.add_argument("--mrest", type=int, default=3, help="CREST --mrest value (default: 3).")
    parser.add_argument("--prop", default="reopt", help="CREST --prop value (default: reopt).")
    parser.add_argument("--no-quick", action="store_true", help="Disable --quick.")

    parser.add_argument(
        "--scratch-root",
        default=None,
        help=(
            "Optional scratch base directory. If set, each molecule gets a dedicated "
            "subdirectory under this root and passed via --scratch."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir).expanduser().resolve()

    sdf_path: Optional[Path] = None
    single_job_dir: Optional[Path] = None

    if args.sdf:
        sdf_path = Path(args.sdf).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        single_job_dir = Path(args.job_dir).expanduser().resolve()

    scratch_root = None
    if args.scratch_root:
        scratch_root = Path(args.scratch_root).expanduser().resolve()
        scratch_root.mkdir(parents=True, exist_ok=True)

    crest_exe_raw = resolve_executable(args.crest_exe, "CREST_EXE", "crest")
    xtb_exe_raw = resolve_executable(args.xtb_exe, "XTB_EXE", "xtb")

    if args.legacy:
        print("[WARN] --legacy is deprecated and ignored in this workflow.")

    crest_exe, crest_err = validate_executable(crest_exe_raw, "CREST")
    if crest_exe is None:
        if args.dry_run:
            crest_exe = crest_exe_raw
            print(f"[WARN] {crest_err}")
            print("[WARN] Continuing because --dry-run does not execute CREST.")
        else:
            print(f"Error: {crest_err}")
            return 2

    xtb_exe, xtb_err = validate_executable(xtb_exe_raw, "xTB")
    if xtb_exe is None:
        if args.dry_run:
            xtb_exe = xtb_exe_raw
            print(f"[WARN] {xtb_err}")
            print("[WARN] Continuing because --dry-run does not execute xTB.")
        else:
            print(f"Error: {xtb_err}")
            return 2

    crest_version = get_crest_version(crest_exe)

    if sdf_path is not None:
        print(f"[INFO] Input SDF: {sdf_path}")
        print(f"[INFO] Output dir: {outdir}")
    else:
        print(f"[INFO] Single job dir: {single_job_dir}")
    print(f"[INFO] CREST exe: {crest_exe}")
    if crest_version:
        print(f"[INFO] CREST version: {crest_version}")
    else:
        print("[INFO] CREST version: unavailable")
    print(f"[INFO] XTB exe: {xtb_exe}")
    print(f"[INFO] Molecules in parallel: {args.max_parallel_molecules}")
    print(f"[INFO] TOML mode: {'enabled' if args.use_toml else 'disabled'}")
    print(f"[INFO] Search method: {args.search_method}")

    summary_rows: List[Dict[str, object]] = []
    prepared_jobs: List[Dict[str, object]] = []

    if sdf_path is not None:
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        processed = 0
        for idx0, mol in enumerate(supplier):
            idx1 = idx0 + 1
            if args.max_mols is not None and processed >= args.max_mols:
                break

            if mol is None:
                summary_rows.append(
                    {
                        "index": idx1,
                        "label": f"mol_{idx1:04d}",
                        "job_dir": "",
                        "status": "skip",
                        "xtb_return_code": "",
                        "crest_return_code": "",
                        "message": "invalid record in SDF",
                    }
                )
                print(f"[WARN] #{idx1:04d}: invalid SDF record; skipped")
                continue

            label = molecule_label(mol, idx1, args.name_prop)
            prepared, prep_error = ensure_3d_coordinates(mol, embed_missing=args.embed_missing)
            if prepared is None:
                summary_rows.append(
                    {
                        "index": idx1,
                        "label": label,
                        "job_dir": "",
                        "status": "skip",
                        "xtb_return_code": "",
                        "crest_return_code": "",
                        "message": prep_error,
                    }
                )
                print(f"[WARN] #{idx1:04d} {label}: {prep_error}; skipped")
                continue

            job_dir = unique_job_dir(outdir, idx1, label)
            job_dir.mkdir(parents=True, exist_ok=False)
            xyz_path = job_dir / "input.xyz"
            Chem.MolToXYZFile(prepared, str(xyz_path), confId=0)

            prepared_jobs.append(
                {
                    "index": idx1,
                    "label": label,
                    "job_dir": str(job_dir),
                }
            )
            processed += 1
            print(f"[INFO] #{idx1:04d} {label}: prepared ({job_dir.name})")
    else:
        job, err = job_from_existing_dir(single_job_dir)
        if job is None:
            print(f"Error: {err}")
            return 2
        prepared_jobs.append(job)

    if prepared_jobs:
        max_workers = max(1, int(args.max_parallel_molecules))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    run_pipeline_for_molecule,
                    job,
                    crest_exe,
                    xtb_exe,
                    args.threads,
                    args.alpb,
                    not args.no_quick,
                    args.ewin,
                    args.rthr,
                    args.mdlen,
                    args.mrest,
                    args.prop,
                    args.use_toml,
                    args.search_method,
                    args.runtype,
                    scratch_root,
                    args.dry_run,
                ): job
                for job in prepared_jobs
            }

            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                summary_rows.append(result)
                print(
                    f"[INFO] #{int(result['index']):04d} {result['label']}: "
                    f"{result['status']} ({result['message']})"
                )

    summary_csv = outdir / "run_summary.csv" if sdf_path is not None else single_job_dir / "run_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "label",
                "job_dir",
                "status",
                "xtb_return_code",
                "crest_return_code",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(summary_rows, key=lambda x: int(x["index"])))

    ok_count = sum(1 for r in summary_rows if r.get("status") == "ok")
    fail_count = sum(1 for r in summary_rows if r.get("status") == "fail")
    skip_count = sum(1 for r in summary_rows if r.get("status") == "skip")

    print("\n[INFO] Run complete")
    print(f"[INFO] Success: {ok_count}, Failed: {fail_count}, Skipped: {skip_count}")
    print(f"[INFO] Summary CSV: {summary_csv}")

    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
