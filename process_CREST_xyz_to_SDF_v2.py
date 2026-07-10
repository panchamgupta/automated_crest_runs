#!/usr/bin/env python3
"""Convert CREST XYZ outputs to SDF using template-first bond assignment.

Two conversion modes:

  template-first (default):
    Uses the input SDF molecule's bond connectivity, formal charges, and bond
    orders directly.  Since CREST preserves atom order from input.xyz (which was
    generated from the SDF via Chem.MolToXYZFile), atom index N in
    crest_conformers.xyz corresponds to atom index N in the original SDF.  No
    graph isomorphism or substructure matching is needed—just direct coordinate
    replacement on a copy of the template molecule.

  infer-from-xyz (fallback):
    Uses RDKit DetermineBonds to perceive connectivity and bond orders from 3D
    coordinates and total formal charge.  This is the legacy behavior and is
    used automatically when no template is available or when atom count/element
    mismatch is detected.

Usage examples:

  # Template-first (recommended)
  python3 process_CREST_xyz_to_SDF_v2.py \\
    --job-list job_dirs.list \\
    --charge-sdf input_molecules.sdf \\
    --charge-id-prop _Name \\
    --mode template-first

  # Legacy infer mode
  python3 process_CREST_xyz_to_SDF_v2.py \\
    --job-list job_dirs.list \\
    --charge-sdf input_molecules.sdf \\
    --charge-id-prop _Name \\
    --mode infer-from-xyz

  # Single file with explicit template
  python3 process_CREST_xyz_to_SDF_v2.py \\
    --xyz crest_conformers.xyz \\
    --template-sdf molecule.sdf \\
    --out crest_conformers.sdf
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

try:
    from rdkit.Chem import rdDetermineBonds
except Exception:
    rdDetermineBonds = None


ENERGY_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")
HARTREE_TO_KCAL_MOL = 627.509
BOND_LENGTH_WARN_THRESHOLD = 3.0  # Angstroms


def bond_signature(mol: Chem.Mol) -> set[tuple[int, int, str]]:
    """Return canonical bond signature: (min_idx, max_idx, bond_type)."""
    sig: set[tuple[int, int, str]] = set()
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        i, j = (a, b) if a < b else (b, a)
        sig.add((i, j, str(bond.GetBondType())))
    return sig


def compare_template_vs_inferred_bonds(
    xyz_path: Path,
    template_mol: Chem.Mol,
    charge: int,
) -> tuple[Optional[bool], str, int, int]:
    """Compare template bond map with XYZ-inferred bond map on first frame.

    Returns:
      (is_same, note, template_bond_count, inferred_bond_count)
    """
    try:
        frames = parse_xyz_frames(xyz_path)
    except Exception as exc:
        return None, f"xyz_parse_error: {exc}", template_mol.GetNumBonds(), -1

    if not frames:
        return None, "no_xyz_frames", template_mol.GetNumBonds(), -1

    first_block, _ = frames[0]

    try:
        inferred = xyz_block_to_mol(first_block, charge=charge)
    except Exception as exc:
        return None, f"infer_failed: {exc}", template_mol.GetNumBonds(), -1

    template_sig = bond_signature(template_mol)
    inferred_sig = bond_signature(inferred)
    is_same = template_sig == inferred_sig

    note = "bond_map_match" if is_same else "bond_map_mismatch"
    return is_same, note, len(template_sig), len(inferred_sig)


# ---------------------------------------------------------------------------
# XYZ parsing
# ---------------------------------------------------------------------------

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
    """Parse a numeric energy from the XYZ comment line."""
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


# ---------------------------------------------------------------------------
# Template lookup builders
# ---------------------------------------------------------------------------

def build_template_lookup_from_sdf(
    sdf_path: Path, id_prop: str = "_Name"
) -> dict[str, Chem.Mol]:
    """Build molecule-id -> RDKit Mol lookup from SDF.

    Returns full molecules with explicit H, bond orders, and formal charges.
    Keys are case-insensitive (lowered). First occurrence wins for duplicates.
    """
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    lookup: dict[str, Chem.Mol] = {}

    for mol in supplier:
        if mol is None:
            continue

        if id_prop == "_Name":
            mol_id = str(mol.GetProp("_Name")).strip() if mol.HasProp("_Name") else ""
        else:
            mol_id = str(mol.GetProp(id_prop)).strip() if mol.HasProp(id_prop) else ""

        if not mol_id:
            continue

        key = mol_id.lower()
        if key in lookup:
            continue

        lookup[key] = mol

    return lookup


def build_charge_lookup_from_sdf(sdf_path: Path, id_prop: str = "_Name") -> dict[str, int]:
    """Build molecule-id -> formal charge lookup from SDF."""
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    lookup: dict[str, int] = {}

    for mol in supplier:
        if mol is None:
            continue

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


# ---------------------------------------------------------------------------
# Mode 1: Template-first conversion
# ---------------------------------------------------------------------------

def xyz_block_to_mol_from_template(xyz_block: str, template_mol: Chem.Mol) -> Chem.Mol:
    """Convert XYZ block to molecule using template bond connectivity.

    Since CREST preserves atom order from the input XYZ (generated from SDF),
    atom index N in the XYZ block corresponds to atom index N in template_mol.

    Steps:
      1. Parse XYZ to get atom coordinates
      2. Validate atom count and element sequence match template
      3. Deep copy template and replace coordinates

    Raises ValueError if atom count or element order does not match.
    """
    # Parse XYZ block to get coordinates
    raw_mol = Chem.MolFromXYZBlock(xyz_block)
    if raw_mol is None:
        raise ValueError("RDKit failed to parse XYZ block")

    # Validate atom count
    n_xyz = raw_mol.GetNumAtoms()
    n_template = template_mol.GetNumAtoms()
    if n_xyz != n_template:
        raise ValueError(
            f"Atom count mismatch: XYZ has {n_xyz} atoms, template has {n_template}"
        )

    # Validate element sequence
    for i in range(n_xyz):
        xyz_elem = raw_mol.GetAtomWithIdx(i).GetSymbol()
        tpl_elem = template_mol.GetAtomWithIdx(i).GetSymbol()
        if xyz_elem != tpl_elem:
            raise ValueError(
                f"Element mismatch at atom {i}: XYZ has {xyz_elem}, template has {tpl_elem}"
            )

    # Deep copy template (preserves bonds, formal charges, stereo, etc.)
    new_mol = Chem.RWMol(template_mol)

    # Extract coordinates from XYZ mol
    xyz_conf = raw_mol.GetConformer(0)
    positions = xyz_conf.GetPositions()  # numpy array (N, 3)

    # Replace coordinates on template copy
    if new_mol.GetNumConformers() == 0:
        conf = Chem.Conformer(n_template)
        new_mol.AddConformer(conf, assignId=True)

    out_conf = new_mol.GetConformer(0)
    for i in range(n_template):
        out_conf.SetAtomPosition(i, positions[i].tolist())

    # Bond-length sanity check (warning only)
    for bond in new_mol.GetBonds():
        idx_a = bond.GetBeginAtomIdx()
        idx_b = bond.GetEndAtomIdx()
        dist = np.linalg.norm(positions[idx_a] - positions[idx_b])
        if dist > BOND_LENGTH_WARN_THRESHOLD:
            print(
                f"  [WARN] Bond {idx_a}-{idx_b} "
                f"({new_mol.GetAtomWithIdx(idx_a).GetSymbol()}-"
                f"{new_mol.GetAtomWithIdx(idx_b).GetSymbol()}) "
                f"length={dist:.2f} Å exceeds {BOND_LENGTH_WARN_THRESHOLD} Å"
            )

    return new_mol.GetMol()


# ---------------------------------------------------------------------------
# Mode 2: Infer-from-XYZ (legacy fallback)
# ---------------------------------------------------------------------------

def xyz_block_to_mol(xyz_block: str, charge: int) -> Chem.Mol:
    """Convert XYZ block to molecule by inferring bonds from 3D coordinates."""
    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise ValueError("RDKit failed to parse XYZ block")

    if rdDetermineBonds is None:
        raise RuntimeError("rdDetermineBonds is unavailable in this RDKit build")

    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    Chem.SanitizeMol(mol)
    return mol


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def to_sentence_case_title(text: str) -> str:
    """Convert folder stem/title to simple sentence case."""
    cleaned = str(text).strip()
    if not cleaned:
        return "Molecule"
    return cleaned[0].upper() + cleaned[1:].lower()


def convert_one_xyz(
    xyz_path: Path,
    out_sdf: Path,
    charge: int,
    base_name: str | None = None,
    energy_prop: str = "CREST_Energy",
    energy_kcal_prop: str = "CREST_Energy_kcalmol",
    relative_energy_prop: str = "CREST_RelativeEnergy_kcalmol",
    comment_prop: str = "CREST_COMMENT",
    template_mol: Optional[Chem.Mol] = None,
) -> Tuple[int, int]:
    """Convert a multi-frame XYZ file to SDF with energy properties.

    If template_mol is provided, uses template-first mode (Mode 1).
    Falls back to infer-from-xyz (Mode 2) if template fails for a frame.
    If template_mol is None, uses Mode 2 exclusively.
    """
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
            mol = None
            if template_mol is not None:
                try:
                    mol = xyz_block_to_mol_from_template(block, template_mol)
                except ValueError as tmpl_exc:
                    if idx == 1:
                        print(
                            f"[WARN] {xyz_path.name}: template-first failed "
                            f"({tmpl_exc}); falling back to infer-from-xyz"
                        )
                    mol = xyz_block_to_mol(block, charge=charge)
            else:
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


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def read_job_dirs(list_path: Path) -> List[Path]:
    """Read job directories from a job_dirs.list file."""
    dirs: List[Path] = []
    list_dir = list_path.parent.resolve()
    cwd = Path.cwd().resolve()
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        p = Path(entry).expanduser()
        if not p.is_absolute():
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
    """Combine multiple SDF files into one."""
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
    template_lookup: Optional[dict[str, Chem.Mol]] = None,
    summary_csv: Optional[Path] = None,
) -> None:
    """Process all job directories from a job list file."""
    job_dirs = read_job_dirs(job_list_path)
    if not job_dirs:
        raise ValueError(f"No valid job directories found in: {job_list_path}")

    total_ok = 0
    total_fail = 0
    output_sdfs: List[Path] = []
    summary_rows: List[dict[str, object]] = []

    for d in job_dirs:
        xyz_path = d / xyz_name
        match_key = d.name.strip().lower()
        if not xyz_path.is_file():
            print(f"[WARN] Missing XYZ in {d.name}: {xyz_path.name}")
            summary_rows.append(
                {
                    "job_dir": d.name,
                    "id_match_found": False,
                    "template_found": False,
                    "charge_lookup_found": False,
                    "charge_used": charge,
                    "bond_map_same_template_vs_inferred": "",
                    "template_bond_count": "",
                    "inferred_bond_count": "",
                    "status": "missing_xyz",
                    "conformers_written": 0,
                    "failed_frames": 0,
                    "note": "missing_xyz",
                }
            )
            continue

        # Resolve charge for this molecule
        charge_to_use = charge
        charge_found = False
        if charge_lookup is not None:
            if match_key in charge_lookup:
                charge_to_use = int(charge_lookup[match_key])
                charge_found = True
            else:
                print(
                    f"[WARN] No charge match for folder '{d.name}' in charge-SDF lookup; "
                    f"falling back to --charge={charge}"
                )

        # Resolve template for this molecule
        template_mol = None
        template_found = False
        if template_lookup is not None:
            if match_key in template_lookup:
                template_mol = template_lookup[match_key]
                template_found = True
            else:
                print(
                    f"[WARN] No template match for folder '{d.name}'; "
                    f"using infer-from-xyz mode"
                )

        # Summary verification: compare template bond map vs XYZ-inferred bond map
        bond_map_same: Optional[bool] = None
        template_bond_count: int | str = ""
        inferred_bond_count: int | str = ""
        note = ""
        if template_mol is not None:
            bond_map_same, note, template_bond_count, inferred_bond_count = compare_template_vs_inferred_bonds(
                xyz_path=xyz_path,
                template_mol=template_mol,
                charge=charge_to_use,
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
            template_mol=template_mol,
        )
        total_ok += ok
        total_fail += fail
        output_sdfs.append(out_sdf)
        print(f"[INFO] {d.name}: wrote {ok} confs to {out_sdf.name}, failed={fail}")

        summary_rows.append(
            {
                "job_dir": d.name,
                "id_match_found": template_found or charge_found,
                "template_found": template_found,
                "charge_lookup_found": charge_found,
                "charge_used": charge_to_use,
                "bond_map_same_template_vs_inferred": "" if bond_map_same is None else str(bond_map_same),
                "template_bond_count": template_bond_count,
                "inferred_bond_count": inferred_bond_count,
                "status": "ok" if fail == 0 else "partial",
                "conformers_written": ok,
                "failed_frames": fail,
                "note": note,
            }
        )

    n_combined = combine_sdfs(output_sdfs, combined_out)
    print(f"[INFO] Combined SDF: {combined_out}")
    print(f"[INFO] Combined conformers written: {n_combined}")
    print(f"[INFO] Total conformers written: {total_ok}")
    print(f"[INFO] Total failed frames: {total_fail}")

    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "job_dir",
            "id_match_found",
            "template_found",
            "charge_lookup_found",
            "charge_used",
            "bond_map_same_template_vs_inferred",
            "template_bond_count",
            "inferred_bond_count",
            "status",
            "conformers_written",
            "failed_frames",
            "note",
        ]
        with summary_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[INFO] Run summary written: {summary_csv}")


def batch_jobs_root(
    jobs_root: Path,
    xyz_name: str,
    sdf_name: str,
    charge: int,
    energy_prop: str,
    energy_kcal_prop: str,
    relative_energy_prop: str,
    comment_prop: str,
    template_lookup: Optional[dict[str, Chem.Mol]] = None,
    charge_lookup: Optional[dict[str, int]] = None,
) -> None:
    """Process all job subfolders under a root directory."""
    job_dirs = sorted([d for d in jobs_root.iterdir() if d.is_dir()])
    total_ok = 0
    total_fail = 0
    n_jobs = 0

    for d in job_dirs:
        xyz_path = d / xyz_name
        if not xyz_path.is_file():
            continue

        n_jobs += 1

        # Resolve charge
        charge_to_use = charge
        if charge_lookup is not None:
            match_key = d.name.strip().lower()
            if match_key in charge_lookup:
                charge_to_use = int(charge_lookup[match_key])

        # Resolve template
        template_mol = None
        if template_lookup is not None:
            match_key = d.name.strip().lower()
            if match_key in template_lookup:
                template_mol = template_lookup[match_key]

        out_sdf = d / sdf_name
        ok, fail = convert_one_xyz(
            xyz_path,
            out_sdf,
            charge=charge_to_use,
            base_name=d.name,
            energy_prop=energy_prop,
            energy_kcal_prop=energy_kcal_prop,
            relative_energy_prop=relative_energy_prop,
            comment_prop=comment_prop,
            template_mol=template_mol,
        )
        total_ok += ok
        total_fail += fail
        print(f"[INFO] {d.name}: wrote {ok} confs to {out_sdf.name}, failed={fail}")

    print(f"[INFO] Jobs processed: {n_jobs}")
    print(f"[INFO] Total conformers written: {total_ok}")
    print(f"[INFO] Total failed frames: {total_fail}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CREST XYZ outputs to SDF (template-first or infer-from-xyz)"
    )
    # Input mode selection
    p.add_argument("--xyz", default=None, help="Single XYZ file to convert")
    p.add_argument("--out", default=None, help="Output SDF path for --xyz mode")
    p.add_argument("--jobs-root", default=None, help="Process all job subfolders under this directory")
    p.add_argument("--job-list", default=None, help="Path to job_dirs.list containing one folder path per line")

    # File naming
    p.add_argument("--xyz-name", default="crest_conformers.xyz", help="XYZ filename inside each job folder")
    p.add_argument("--sdf-name", default="crest_conformers.sdf", help="Output SDF filename inside each job folder")
    p.add_argument("--combined-out", default=None, help="Combined output SDF path for --job-list mode")
    p.add_argument(
        "--summary-csv",
        default=None,
        help="Per-molecule verification summary CSV for --job-list mode "
        "(default: <job_list_dir>/process_CREST_xyz_to_SDF_v2_run_summary.csv)",
    )

    # Bond assignment mode
    p.add_argument(
        "--mode", choices=["template-first", "infer-from-xyz"], default="template-first",
        help="Bond assignment mode (default: template-first)"
    )

    # Template / charge source
    p.add_argument(
        "--charge-sdf", default=None,
        help="Input SDF providing template molecules (template-first mode) and/or formal charges (infer-from-xyz mode)"
    )
    p.add_argument("--charge-id-prop", default="_Name", help="SDF property for molecule ID matching (default: _Name)")
    p.add_argument("--charge", type=int, default=0, help="Fallback net molecular charge (default: 0)")

    # For single-file mode with explicit template
    p.add_argument("--template-sdf", default=None, help="Explicit template SDF for --xyz single-file mode")

    # Energy property names
    p.add_argument("--energy-prop", default="CREST_Energy", help="SDF property for energy in Hartree")
    p.add_argument("--energy-kcal-prop", default="CREST_Energy_kcalmol", help="SDF property for energy in kcal/mol")
    p.add_argument("--relative-energy-prop", default="CREST_RelativeEnergy_kcalmol", help="SDF property for relative energy")
    p.add_argument("--comment-prop", default="CREST_COMMENT", help="SDF property for XYZ comment line")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    use_template = args.mode == "template-first"

    # --job-list mode
    if args.job_list:
        job_list_path = Path(args.job_list).expanduser().resolve()
        combined_out = (
            Path(args.combined_out).expanduser().resolve()
            if args.combined_out
            else (job_list_path.parent / "crest_jobs_combined.sdf")
        )
        summary_csv = (
            Path(args.summary_csv).expanduser().resolve()
            if args.summary_csv
            else (job_list_path.parent / "process_CREST_xyz_to_SDF_v2_run_summary.csv")
        )

        template_lookup = None
        charge_lookup = None

        if args.charge_sdf:
            charge_sdf_path = Path(args.charge_sdf).expanduser().resolve()
            if use_template:
                template_lookup = build_template_lookup_from_sdf(
                    charge_sdf_path, id_prop=args.charge_id_prop
                )
                print(
                    f"[INFO] Loaded {len(template_lookup)} template molecules "
                    f"from {charge_sdf_path} (template-first mode)"
                )
            # Always build charge lookup as fallback
            charge_lookup = build_charge_lookup_from_sdf(
                charge_sdf_path, id_prop=args.charge_id_prop
            )
            print(f"[INFO] Loaded {len(charge_lookup)} charge entries (fallback)")

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
            template_lookup,
            summary_csv,
        )
        return 0

    # --jobs-root mode
    if args.jobs_root:
        template_lookup = None
        charge_lookup = None

        if args.charge_sdf:
            charge_sdf_path = Path(args.charge_sdf).expanduser().resolve()
            if use_template:
                template_lookup = build_template_lookup_from_sdf(
                    charge_sdf_path, id_prop=args.charge_id_prop
                )
                print(
                    f"[INFO] Loaded {len(template_lookup)} template molecules "
                    f"from {charge_sdf_path} (template-first mode)"
                )
            charge_lookup = build_charge_lookup_from_sdf(
                charge_sdf_path, id_prop=args.charge_id_prop
            )

        batch_jobs_root(
            Path(args.jobs_root).expanduser().resolve(),
            args.xyz_name,
            args.sdf_name,
            args.charge,
            args.energy_prop,
            args.energy_kcal_prop,
            args.relative_energy_prop,
            args.comment_prop,
            template_lookup,
            charge_lookup,
        )
        return 0

    # --xyz single-file mode
    if args.xyz:
        xyz_path = Path(args.xyz).expanduser().resolve()
        out_sdf = Path(args.out).expanduser().resolve() if args.out else xyz_path.with_suffix(".sdf")

        template_mol = None
        if use_template:
            # Use --template-sdf or first molecule from --charge-sdf
            tpl_path = None
            if args.template_sdf:
                tpl_path = Path(args.template_sdf).expanduser().resolve()
            elif args.charge_sdf:
                tpl_path = Path(args.charge_sdf).expanduser().resolve()

            if tpl_path and tpl_path.is_file():
                supplier = Chem.SDMolSupplier(str(tpl_path), removeHs=False)
                for mol in supplier:
                    if mol is not None:
                        template_mol = mol
                        break
                if template_mol is not None:
                    print(f"[INFO] Using template from {tpl_path} (template-first mode)")
                else:
                    print(f"[WARN] No valid molecule in {tpl_path}; falling back to infer-from-xyz")

        ok, fail = convert_one_xyz(
            xyz_path,
            out_sdf,
            charge=args.charge,
            energy_prop=args.energy_prop,
            energy_kcal_prop=args.energy_kcal_prop,
            relative_energy_prop=args.relative_energy_prop,
            comment_prop=args.comment_prop,
            template_mol=template_mol,
        )
        print(f"[INFO] Wrote {ok} conformers to {out_sdf}")
        print(f"[INFO] Failed frames: {fail}")
        return 0

    print("Error: provide one of --xyz, --jobs-root, or --job-list")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
