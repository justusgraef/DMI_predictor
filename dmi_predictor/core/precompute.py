"""
Feature precomputation workflow for DMI Predictor.

This module generates per-protein features (IUPred, Anchor, DomainOverlap) for all input FASTA sequences.
Requires AIUPred (deep learning) for disorder prediction.
"""
from pathlib import Path
from typing import Optional, List, Tuple

# AIUPred integration (replaces IUPred2a with modern deep learning)
try:
    import aiupred_lib
    HAS_AIUPRED = True
except ImportError:
    HAS_AIUPRED = False


def _load_fasta_from_dir(fasta_dir: Path) -> List[Tuple[str, str]]:
    sequences: List[Tuple[str, str]] = []
    fasta_files = list(fasta_dir.glob("*.fasta"))
    for fasta_file in fasta_files:
        protein_id = None
        seq_parts: List[str] = []
        with open(fasta_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):  # header
                    protein_id = line[1:].split()[0]
                else:
                    seq_parts.append(line)
        if protein_id and seq_parts:
            sequences.append((protein_id, "".join(seq_parts)))
    return sequences


def _load_fasta_from_file(fasta_file: Path) -> List[Tuple[str, str]]:
    sequences: List[Tuple[str, str]] = []
    current_id = None
    current_seq: List[str] = []
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    sequences.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id:
        sequences.append((current_id, "".join(current_seq)))
    return sequences


def precompute_features(
    fasta_dir: Optional[str] = None,
    output_dir: str = "features",
    verbose: bool = False,
    force_cpu: bool = False,
    fasta_file: Optional[str] = None,
) -> None:
    if (fasta_dir is None and fasta_file is None) or (fasta_dir and fasta_file):
        print("Provide exactly one of --fasta-dir or --fasta-file")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    iupred_dir = output_dir / "IUPred_short"
    anchor_dir = output_dir / "Anchor"
    domain_overlap_dir = output_dir / "Domain_overlap"
    for subdir in (iupred_dir, anchor_dir, domain_overlap_dir):
        subdir.mkdir(parents=True, exist_ok=True)

    if fasta_dir:
        sequences = _load_fasta_from_dir(Path(fasta_dir))
    else:
        sequences = _load_fasta_from_file(Path(fasta_file))
    if not sequences:
        print("No FASTA sequences found")
        return
    
    # AIUPred is required
    embedding_model = None
    regression_model = None
    device = None
    if not HAS_AIUPRED:
        raise RuntimeError("AIUPred is required for precompute-features. Install aiupred-lib and set PYTHONPATH.")

    try:
        if verbose:
            print("Loading AIUPred models...")
        embedding_model, regression_model, device = aiupred_lib.init_models(
            'disorder', force_cpu=force_cpu
        )
    except Exception as e:
        raise RuntimeError(f"Could not initialize AIUPred: {e}")

    for protein_id, sequence in sequences:
        if verbose:
            print(f"Precomputing features for {protein_id}")
        # Disorder (IUPred_short) and Anchor features via AIUPred
        iupred_short = aiupred_lib.predict_disorder(
            sequence, embedding_model, regression_model, device, smoothing=True
        )
        # AIUPred doesn't natively predict anchors; use disorder-based proxy
        anchor = [0.5 - abs(0.5 - score) for score in iupred_short]  # Inverted disorder
        # DomainOverlap (dummy for now)
        domain_overlap = [0] * len(sequence)
        # Write features to output_dir
        with open(iupred_dir / f"{protein_id}_iupredshort.txt", "w") as f:
            f.write(f"{protein_id}\n")
            for i, score in enumerate(iupred_short):
                f.write(f"{i+1}\t{sequence[i]}\t{score}\n")
        with open(anchor_dir / f"{protein_id}_anchor.txt", "w") as f:
            f.write(f"{protein_id}\n")
            for i, score in enumerate(anchor):
                f.write(f"{i+1}\t{sequence[i]}\t{score}\n")
        with open(domain_overlap_dir / f"{protein_id}_domain_overlap.txt", "w") as f:
            f.write(f"{protein_id}\n")
            for i, score in enumerate(domain_overlap):
                f.write(f"{i+1}\t{sequence[i]}\t{score}\n")
        if verbose:
            print(f"Features written for {protein_id}")
