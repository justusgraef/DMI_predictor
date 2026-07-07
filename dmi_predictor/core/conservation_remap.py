"""
Conservation-score position re-alignment.

The conservation-score library stores per-residue RLC scores keyed by ABSOLUTE
sequence position, computed against the sequences that were canonical when the
library was built. When a protein's UniProt sequence is later updated (e.g. an
isoform change that adds N-terminal residues), position N no longer refers to the
same residue, so ``predict`` reads misaligned conservation and mis-scores that
protein's DMIs.

This is the conservation-score analogue of :mod:`domain_correction`. Like that
module it RE-USES existing data rather than recomputing: each score is relocated
to its new coordinate via an old->new sequence alignment. Scores are exact for
residues shared between the old and new sequence; residues that exist only in the
new sequence (e.g. a new N-terminus) are left unscored and fall back to median
imputation downstream — exactly how the pipeline already treats any protein or
region lacking conservation data.

It re-uses existing scores; it does NOT regenerate conservation from orthologs.
For genuinely new residues, rerun the original conservation pipeline.

Original files are never modified — corrected copies are written to ``out_dir``,
which callers point the predictor at (it takes precedence over the source library).
"""

import difflib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence


def read_fasta(path: str) -> Dict[str, str]:
    """Return {accession: sequence}; accession = first whitespace token of header."""
    seqs: Dict[str, str] = {}
    acc, buf = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None:
                    seqs[acc] = "".join(buf)
                acc = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if acc is not None:
        seqs[acc] = "".join(buf)
    return seqs


def build_position_map(old_seq: str, new_seq: str) -> Dict[int, int]:
    """
    Map 1-based old positions -> 1-based new positions for residues that are
    identical and co-located under a longest-matching-block alignment.

    Uses difflib matching blocks, which cleanly handles terminal extensions/trims
    and internal indels. A score is transferred only when the residue is unchanged.
    """
    sm = difflib.SequenceMatcher(a=old_seq, b=new_seq, autojunk=False)
    pos_map: Dict[int, int] = {}
    for i, j, n in sm.get_matching_blocks():  # old[i:i+n] == new[j:j+n]
        for k in range(n):
            pos_map[i + k + 1] = j + k + 1  # 0-based -> 1-based
    return pos_map


def remap_conservation_obj(cons_obj: dict, pos_map: Dict[int, int]) -> dict:
    """
    Return a new Conservation object with every level's position keys remapped
    through ``pos_map``. Old positions with no new counterpart are dropped; new
    positions with no old counterpart are simply never created.
    """
    new_conservation = []
    for entry in cons_obj["Conservation"]:
        # each entry is a single-key dict: {level: {pos_str: score}}
        (level, scores), = entry.items()
        remapped = {}
        for pos_str, score in scores.items():
            new_pos = pos_map.get(int(pos_str))
            if new_pos is not None:
                remapped[str(new_pos)] = score
        new_conservation.append({level: remapped})
    return {"Conservation": new_conservation}


def run_conservation_remap(
    reference_fasta: str,
    current_sequences: Dict[str, str],
    cons_dir: str,
    out_dir: str,
    proteins: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> int:
    """
    Re-align conservation-score files to updated sequences and write corrected
    copies to ``out_dir``.

    Args:
        reference_fasta: FASTA the conservation library was built against (old sequences).
        current_sequences: {accession: sequence} for the current prediction run.
        cons_dir: directory of existing ``<UniProt>_con.json`` files.
        out_dir: directory to write corrected ``<UniProt>_con.json`` files into.
        proteins: optional explicit accessions; default = auto-detect changed sequences.
        verbose: print per-protein progress.

    Returns:
        Number of proteins whose conservation file was corrected.
    """
    reference = read_fasta(reference_fasta)
    cons_path_dir = Path(cons_dir)
    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    # Which proteins to process: those present in both sets whose sequence changed.
    if proteins:
        targets = list(proteins)
    else:
        common = set(reference) & set(current_sequences)
        targets = sorted(p for p in common if reference[p] != current_sequences[p])

    if verbose:
        print(
            f"Conservation remap: {len(targets)} changed sequence(s) to re-align "
            f"({', '.join(targets) if targets else '—'})"
        )

    n_corrected = 0
    for pid in targets:
        if pid not in reference or pid not in current_sequences:
            if verbose:
                print(f"  {pid}: skip — not in both reference and current sequences")
            continue
        src = cons_path_dir / f"{pid}_con.json"
        if not src.exists():
            if verbose:
                print(f"  {pid}: skip — no conservation file at {src}")
            continue

        old_seq, new_seq = reference[pid], current_sequences[pid]
        with open(src) as f:
            cons_obj = json.load(f)

        # Guard: conservation should be indexed 1..len(old_seq); warn if not.
        max_pos = max(
            (int(k) for entry in cons_obj["Conservation"]
             for k in next(iter(entry.values())).keys()),
            default=0,
        )
        if verbose and max_pos != len(old_seq):
            print(
                f"  {pid}: [warn] conservation max position ({max_pos}) != reference "
                f"sequence length ({len(old_seq)}); alignment reference may be off"
            )

        pos_map = build_position_map(old_seq, new_seq)
        corrected = remap_conservation_obj(cons_obj, pos_map)

        with open(out_path_dir / f"{pid}_con.json", "w") as f:
            json.dump(corrected, f)

        n_corrected += 1
        if verbose:
            n_mapped = len(next(iter(corrected["Conservation"][0].values())))
            unscored = len(new_seq) - n_mapped
            print(
                f"  {pid}: old={len(old_seq)} new={len(new_seq)}  "
                f"mapped={n_mapped}  unscored(new-only)={unscored}"
            )

    print(f"Conservation remap complete: {n_corrected} file(s) re-aligned into {out_path_dir}")
    return n_corrected
