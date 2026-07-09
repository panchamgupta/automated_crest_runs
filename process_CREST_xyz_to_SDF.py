#!/usr/bin/env python3
"""Convert CREST XYZ outputs to SDF.

Supports:
1) Single file mode:
   python3 crest_xyz_to_sdf.py --xyz crest_conformers.xyz --out crest_conformers.sdf

2) Batch jobs-root mode:
   python3 crest_xyz_to_sdf.py --jobs-root crest_jobs

cd /home/pgupta11/Projects/B3GNT2/macrocycles/program_writing/crest
/home/pgupta11/anaconda3/envs/rdkit-env/bin/python crest_xyz_to_sdf.py \
  --jobs-root crest_jobs \
  --xyz-name crest_conformers.xyz \
  --sdf-name crest_conformers.sdf

cd /home/pgupta11/Projects/B3GNT2/macrocycles/program_writing/crest
/home/pgupta11/anaconda3/envs/rdkit-env/bin/python crest_xyz_to_sdf.py \
  --xyz crest_jobs/gs-1701253/crest_conformers.xyz \
  --out crest_jobs/gs-1701253/crest_conformers.sdf
  
Notes:
- CREST XYZ files do not contain bond topology. This script uses RDKit bond
  perception (DetermineBonds) for each XYZ frame.
- For charged systems, pass --charge to improve bond assignment.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem

try:
    from rdkit.Chem import rdDetermineBonds
except Exception:
    rdDetermineBonds = None


ENERGY_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")
HARTREE_TO_KCAL_MOL = 627.509


def parse_xyz_frames(xyz_path: Path) -> List[Tuple[str, str]]:
    """Parse concatenated XYZ into [(xyz_block, comment), ...]."""
    lines = xyz_path.read_text(encoding="utf-8", errors="replace").splitlines()
    frames: List[Tuple[str, str]] = []
    i = 0

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        try:
            natoms = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ atom-count line at {i+1} in {xyz_path}") from exc

        if i + 1 >= len(lines):
            raise ValueError(f"Missing XYZ comment line after atom count at {i+1}")

        comment = lines[i + 1].rstrip("\n")
        start = i + 2
        end = start + natoms
        if end > len(lines):
            raise ValueError(f"Unexpected EOF while reading frame near line {i+1}")

        atom_lines = lines[start:end]
        xyz_block = "\n".join([str(natoms), comment] + atom_lines) + "\n"
        frames.append((xyz_block, comment))
        i = end

    return frames


def parse_energy_from_comment(comment: str) -> Optional[float]:
    """Parse a numeric energy from the XYZ comment line.

    CREST commonly writes the energy on the second line of each frame.
    If the comment contains extra text, the first parseable floating-point
    token is used.
    """
    text = comment.strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        match = ENERGY_RE.search(text)
        if match is None:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None


def to_sentence_case_title(text: str) -> str:
    """Convert folder stem/title to simple sentence case."""
    cleaned = str(text).strip()
    if not cleaned:
        return "Molecule"
    return cleaned[0].upper() + cleaned[1:].lower()


def build_charge_lookup_from_sdf(sdf_path: Path, id_prop: str = "_Name") -> dict[str, int]:
    """Build molecule-id -> formal charge lookup from SDF.

    Matching is case-insensitive in downstream lookup. If duplicate IDs are
    present, the first occurrence is used.
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    lookup: dict[str, int] = {}

    for i, mol in enumerate(supplier, start=1):
        if mol is None:
            continue

        mol_id = ""
        if id_prop == "_Name":
            mol_id = str(mol.GetProp("_Name")).strip() if mol.HasProp("_Name") else ""
        else:
            mol_id = str(mol.GetProp(id_prop)).strip() if mol.HasProp(id_prop) else ""

        if not mol_id:
            continue

        key = mol_id.lower()
        if key in lookup:
            continue

        formal_charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
        lookup[key] = formal_charge

    return lookup


def xyz_block_to_mol(xyz_block: str, charge: int) -> Chem.Mol:
    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise ValueError("RDKit failed to parse XYZ block")

    if rdDetermineBonds is None:
        raise RuntimeError("rdDetermineBonds is unavailable in this RDKit build")

    # Determine connectivity and bond orders from 3D coordinates.
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    Chem.SanitizeMol(mol)
    return mol


def convert_one_xyz(
    xyz_path: Path,
    out_sdf: Path,
    charge: int,
    base_name: str | None = None,
    energy_prop: str = "CREST_Energy",
    energy_kcal_prop: str = "CREST_Energy_kcalmol",
    relative_energy_prop: str = "CREST_RelativeEnergy_kcalmol",
    comment_prop: str = "CREST_COMMENT",
) -> Tuple[int, int]:
    frames = parse_xyz_frames(xyz_path)
    out_sdf.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    writer = Chem.SDWriter(str(out_sdf))
    stem = base_name if base_name else xyz_path.stem

    parsed_frames = []
    energy_values = []
    for idx, (block, comment) in enumerate(frames, start=1):
        parsed_frames.append((idx, block, comment, parse_energy_from_comment(comment)))
        if parsed_frames[-1][3] is not None:
            energy_values.append(parsed_frames[-1][3])

    min_energy = min(energy_values) if energy_values else None

    for idx, block, comment, energy_hartree in parsed_frames:
        try:
            mol = xyz_block_to_mol(block, charge=charge)
            mol.SetProp("_Name", f"{stem}_conf_{idx:04d}")
            mol.SetProp(comment_prop, comment)
            if energy_hartree is not None:
                energy_kcal = energy_hartree * HARTREE_TO_KCAL_MOL
                mol.SetProp(energy_prop, f"{energy_hartree:.10f}")
                mol.SetProp(energy_kcal_prop, f"{energy_kcal:.4f}")
                if min_energy is not None:
                    rel_kcal = (energy_hartree - min_energy) * HARTREE_TO_KCAL_MOL
                    mol.SetProp(relative_energy_prop, f"{rel_kcal:.4f}")
            writer.write(mol)
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"[WARN] {xyz_path.name} frame {idx}: {exc}")

    writer.close()
    return n_ok, n_fail


def batch_jobs_root(
    jobs_root: Path,
    xyz_name: str,
    sdf_name: str,
    charge: int,
    energy_prop: str,
    energy_kcal_prop: str,
    relative_energy_prop: str,
    comment_prop: str,
) -> None:
    job_dirs = sorted([d for d in jobs_root.iterdir() if d.is_dir()])
    total_ok = 0
    total_fail = 0
    n_jobs = 0

    for d in job_dirs:
        xyz_path = d / xyz_name
        if not xyz_path.is_file():
            continue

        n_jobs += 1
        out_sdf = d / sdf_name
        ok, fail = convert_one_xyz(
            xyz_path,
            out_sdf,
            charge=charge,
            base_name=d.name,
            energy_prop=energy_prop,
            energy_kcal_prop=energy_kcal_prop,
            relative_energy_prop=relative_energy_prop,
            comment_prop=comment_prop,
        )
        total_ok += ok
        total_fail += fail
        print(f"[INFO] {d.name}: wrote {ok} confs to {out_sdf.name}, failed={fail}")

    print(f"[INFO] Jobs processed: {n_jobs}")
    print(f"[INFO] Total conformers written: {total_ok}")
    print(f"[INFO] Total failed frames: {total_fail}")


def read_job_dirs(list_path: Path) -> List[Path]:
    dirs: List[Path] = []
    list_dir = list_path.parent.resolve()
    cwd = Path.cwd().resolve()
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        p = Path(entry).expanduser()
        if not p.is_absolute():
            # Prefer paths relative to job_dirs.list location.
            cand1 = (list_dir / p).resolve()
            cand2 = (cwd / p).resolve()
            if cand1.is_dir():
                p = cand1
            else:
                p = cand2
        if p.is_dir():
            dirs.append(p)
        else:
            print(f"[WARN] Skipping missing/non-directory entry in job list: {entry}")
    return dirs


def combine_sdfs(sdf_paths: List[Path], combined_out: Path) -> int:
    combined_out.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(combined_out))
    n_written = 0

    for sdf_path in sdf_paths:
        if not sdf_path.is_file() or sdf_path.stat().st_size == 0:
            continue
        try:
            supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
            for mol in supplier:
                if mol is None:
                    continue
                writer.write(mol)
                n_written += 1
        except Exception as exc:
            print(f"[WARN] Skipping invalid SDF file {sdf_path}: {exc}")
            continue

    writer.close()
    return n_written


def convert_from_job_list(
    job_list_path: Path,
    xyz_name: str,
    sdf_name: str,
    charge: int,
    energy_prop: str,
    energy_kcal_prop: str,
    relative_energy_prop: str,
    comment_prop: str,
    combined_out: Path,
    charge_lookup: Optional[dict[str, int]] = None,
) -> None:
    job_dirs = read_job_dirs(job_list_path)
    if not job_dirs:
        raise ValueError(f"No valid job directories found in: {job_list_path}")

    total_ok = 0
    total_fail = 0
    output_sdfs: List[Path] = []

    for d in job_dirs:
        xyz_path = d / xyz_name
        if not xyz_path.is_file():
            print(f"[WARN] Missing XYZ in {d.name}: {xyz_path.name}")
            continue

        charge_to_use = charge
        if charge_lookup is not None:
            match_key = d.name.strip().lower()
            if match_key in charge_lookup:
                charge_to_use = int(charge_lookup[match_key])
            else:
                print(
                    f"[WARN] No charge match for folder '{d.name}' in charge-SDF lookup; "
                    f"falling back to --charge={charge}"
                )

        out_sdf = d / sdf_name
        title = to_sentence_case_title(d.name)
        ok, fail = convert_one_xyz(
            xyz_path,
            out_sdf,
            charge=charge_to_use,
            base_name=title,
            energy_prop=energy_prop,
            energy_kcal_prop=energy_kcal_prop,
            relative_energy_prop=relative_energy_prop,
            comment_prop=comment_prop,
        )
        total_ok += ok
        total_fail += fail
        output_sdfs.append(out_sdf)
        print(f"[INFO] {d.name}: wrote {ok} confs to {out_sdf.name}, failed={fail}")

    n_combined = combine_sdfs(output_sdfs, combined_out)
    print(f"[INFO] Combined SDF: {combined_out}")
    print(f"[INFO] Combined conformers written: {n_combined}")
    print(f"[INFO] Total conformers written: {total_ok}")
    print(f"[INFO] Total failed frames: {total_fail}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert CREST XYZ outputs to SDF")
    p.add_argument("--xyz", default=None, help="Single XYZ file to convert")
    p.add_argument("--out", default=None, help="Output SDF path for --xyz mode")
    p.add_argument("--jobs-root", default=None, help="Process all job subfolders under this directory")
    p.add_argument("--job-list", default=None, help="Path to job_dirs.list containing one folder path per line")
    p.add_argument("--xyz-name", default="crest_conformers.xyz", help="XYZ filename inside each job folder")
    p.add_argument("--sdf-name", default="crest_conformers.sdf", help="Output SDF filename inside each job folder")
    p.add_argument("--combined-out", default=None, help="Combined output SDF path for --job-list mode (default: <job_list_dir>/crest_jobs_combined.sdf)")
    p.add_argument("--charge", type=int, default=0, help="Net molecular charge for bond perception (default: 0)")
    p.add_argument("--charge-sdf", default=None, help="Optional SDF used to derive per-folder formal charges for --job-list mode")
    p.add_argument("--charge-id-prop", default="_Name", help="SDF property used as molecule ID for folder-to-charge matching (default: _Name)")
    p.add_argument("--energy-prop", default="CREST_Energy", help="SDF property name for parsed energy")
    p.add_argument("--energy-kcal-prop", default="CREST_Energy_kcalmol", help="SDF property name for converted energy in kcal/mol")
    p.add_argument("--relative-energy-prop", default="CREST_RelativeEnergy_kcalmol", help="SDF property name for relative energy in kcal/mol")
    p.add_argument("--comment-prop", default="CREST_COMMENT", help="SDF property name for original XYZ comment")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.job_list:
        job_list_path = Path(args.job_list).expanduser().resolve()
        combined_out = (
            Path(args.combined_out).expanduser().resolve()
            if args.combined_out
            else (job_list_path.parent / "crest_jobs_combined.sdf")
        )
        charge_lookup = None
        if args.charge_sdf:
            charge_sdf_path = Path(args.charge_sdf).expanduser().resolve()
            charge_lookup = build_charge_lookup_from_sdf(charge_sdf_path, id_prop=args.charge_id_prop)
            print(f"[INFO] Loaded {len(charge_lookup)} unique ID->charge entries from {charge_sdf_path}")
        convert_from_job_list(
            job_list_path,
            args.xyz_name,
            args.sdf_name,
            args.charge,
            args.energy_prop,
            args.energy_kcal_prop,
            args.relative_energy_prop,
            args.comment_prop,
            combined_out,
            charge_lookup,
        )
        return 0

    if args.jobs_root:
        batch_jobs_root(
            Path(args.jobs_root).expanduser().resolve(),
            args.xyz_name,
            args.sdf_name,
            args.charge,
            args.energy_prop,
            args.energy_kcal_prop,
            args.relative_energy_prop,
            args.comment_prop,
        )
        return 0

    if args.xyz:
        xyz_path = Path(args.xyz).expanduser().resolve()
        out_sdf = Path(args.out).expanduser().resolve() if args.out else xyz_path.with_suffix(".sdf")
        ok, fail = convert_one_xyz(
            xyz_path,
            out_sdf,
            charge=args.charge,
            energy_prop=args.energy_prop,
            energy_kcal_prop=args.energy_kcal_prop,
            relative_energy_prop=args.relative_energy_prop,
            comment_prop=args.comment_prop,
        )
        print(f"[INFO] Wrote {ok} conformers to {out_sdf}")
        print(f"[INFO] Failed frames: {fail}")
        return 0

    print("Error: provide one of --xyz, --jobs-root, or --job-list")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
