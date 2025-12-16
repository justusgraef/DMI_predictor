"""
Feature precomputation workflow for DMI Predictor.

This module generates per-protein features (IUPred, Anchor, DomainOverlap) for all input FASTA sequences.
Requires AIUPred (deep learning) or IUPred2A for disorder prediction.
"""
from pathlib import Path
from typing import Optional, List, Tuple
import json
import concurrent.futures

from dmi_predictor.config import DMIPredictorConfig

# AIUPred integration (deep learning disorder predictor)
try:
    import aiupred_lib
    HAS_AIUPRED = True
except ImportError:
    HAS_AIUPRED = False

# IUPred2A integration (original IUPred + ANCHOR2)
try:
    import iupred2a_lib
    HAS_IUPRED2A = True
except ImportError:
    HAS_IUPRED2A = False


def _domain_overlap_scores(
    protein_id: str,
    seq_len: int,
    smart_domain_matches,
    pfam_domain_matches,
    motif_disordered_hmms,
):
    """Compute binary domain overlap vector using provided SMART/Pfam matches."""
    if smart_domain_matches is None and pfam_domain_matches is None:
        return None
    scores = [0] * seq_len

    def apply_matches(db_obj):
        try:
            for result in db_obj.get('results', []):
                if result.get('metadata', {}).get('accession') == protein_id:
                    for entry in result.get('entry_subset', []):
                        entry_accession = entry.get('accession', '')
                        # Skip if this HMM is marked as Motif/Disordered
                        if entry_accession in motif_disordered_hmms:
                            if motif_disordered_hmms[entry_accession] != 'Motif':
                                # It's Disordered (Pfam) or Motif (SMART), skip
                                continue
                        for loc in entry.get('entry_protein_locations', []):
                            for frag in loc.get('fragments', []):
                                start = int(frag.get('start', 1))
                                end = int(frag.get('end', 0))
                                if start < 1:
                                    start = 1
                                if end > seq_len:
                                    end = seq_len
                                for i in range(start - 1, end):
                                    scores[i] = 1
        except Exception:
            pass

    if smart_domain_matches is not None:
        apply_matches(smart_domain_matches)
    if pfam_domain_matches is not None:
        apply_matches(pfam_domain_matches)
    return scores


def _process_protein_task(args):
    (
        protein_id,
        sequence,
        iupred_backend,
        allow_missing_aiupred,
        force_cpu,
        verbose,
        iupred_dir,
        anchor_dir,
        domain_overlap_dir,
        smart_domain_matches,
        pfam_domain_matches,
        motif_disordered_hmms,
    ) = args

    # Compute disorder/anchor
    if iupred_backend == "aiupred":
        if not HAS_AIUPRED:
            if not allow_missing_aiupred:
                raise RuntimeError(
                    "AIUPred backend selected but AIUPred is not installed. Use --allow-missing-aiupred to skip"
                )
            if verbose:
                print(f"[{protein_id}] AIUPred missing; skipping disorder features")
        else:
            embedding_model, regression_model, device = aiupred_lib.init_models('disorder', force_cpu=force_cpu)
            iupred_short = aiupred_lib.predict_disorder(
                sequence, embedding_model, regression_model, device, smoothing=True
            )
            with open(iupred_dir / f"{protein_id}_iupredshort.txt", "w") as f:
                f.write(f"{protein_id}\n")
                for i, score in enumerate(iupred_short):
                    f.write(f"{i+1}\t{sequence[i]}\t{score}\n")
            if verbose:
                print(f"[{protein_id}] AIUPred features written")

    elif iupred_backend == "iupred2a":
        if not HAS_IUPRED2A:
            raise RuntimeError(
                "IUPred2A backend selected but iupred2a_lib not available. Ensure iupred2a_lib is importable."
            )
        result_iupredshort = iupred2a_lib.iupred(sequence, mode='short')[0]
        with open(iupred_dir / f"{protein_id}_iupredshort.txt", "w") as f:
            f.write(f"{protein_id}\n")
            for pos, residue in enumerate(sequence):
                f.write(f"{pos+1}\t{residue}\t{result_iupredshort[pos]}\n")

        result_anchor = iupred2a_lib.anchor2(sequence)
        with open(anchor_dir / f"{protein_id}_anchor.txt", "w") as f:
            f.write(f"{protein_id}\n")
            for pos, residue in enumerate(sequence):
                f.write(f"{pos+1}\t{residue}\t{result_anchor[pos]}\n")

        if verbose:
            print(f"[{protein_id}] IUPred2A + Anchor written")

    else:
        raise ValueError(f"Unknown iupred_backend: {iupred_backend}")

    # Domain overlap
    dom_scores = _domain_overlap_scores(
        protein_id,
        len(sequence),
        smart_domain_matches,
        pfam_domain_matches,
        motif_disordered_hmms,
    )
    if dom_scores is not None:
        with open(domain_overlap_dir / f"{protein_id}_domain_overlap.txt", "w") as f:
            f.write(f"{protein_id}\n")
            for pos, residue in enumerate(sequence):
                f.write(f"{pos+1}\t{residue}\t{dom_scores[pos]}\n")
        if verbose:
            print(f"[{protein_id}] DomainOverlap written")
    else:
        if verbose:
            print(f"[{protein_id}] SMART/Pfam domain JSONs not found — DomainOverlap will be imputed downstream.")

    return protein_id


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
    allow_missing_aiupred: bool = False,
    write_placeholders: bool = False,
    iupred_backend: str = "aiupred",
    num_workers: int = 1,
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

    # Load domain match JSONs (SMART/Pfam) and motif/disordered HMM exclusions for DomainOverlap
    cfg = DMIPredictorConfig()
    data_dir = cfg.data_dir
    smart_json_path = data_dir / "interpro_9606_smart_matches_20210122.json"
    pfam_json_path = data_dir / "interpro_9606_pfam_matches_20210122.json"
    hmm_exclusions_path = data_dir / "motif_disordered_smart_pfam_hmms.json"
    smart_domain_matches = None
    pfam_domain_matches = None
    motif_disordered_hmms = {}
    
    if smart_json_path.exists():
        try:
            with open(smart_json_path, 'r') as f:
                smart_domain_matches = json.load(f)
        except Exception:
            smart_domain_matches = None
    if pfam_json_path.exists():
        try:
            with open(pfam_json_path, 'r') as f:
                pfam_domain_matches = json.load(f)
        except Exception:
            pfam_domain_matches = None
    if hmm_exclusions_path.exists():
        try:
            with open(hmm_exclusions_path, 'r') as f:
                motif_disordered_hmms = json.load(f)
        except Exception:
            motif_disordered_hmms = {}

    if fasta_dir:
        sequences = _load_fasta_from_dir(Path(fasta_dir))
    else:
        sequences = _load_fasta_from_file(Path(fasta_file))
    if not sequences:
        print("No FASTA sequences found")
        return
    
    # AIUPred availability check (models are initialized per worker)
    if iupred_backend == "aiupred":
        if not HAS_AIUPRED:
            msg = (
                "AIUPred not available: precompute can only produce IUPred disorder "
                "features when AIUPred is installed.\n"
                "To install AIUPred (recommended):\n"
                "  git clone https://github.com/doszilab/AIUPred.git /tmp/AIUPred\n"
                "  export PYTHONPATH=/tmp/AIUPred:$PYTHONPATH\n"
                "Alternatively, run with --allow-missing-aiupred to skip AIUPred and "
                "leave disorder features missing (they will be imputed downstream)."
            )
            if not allow_missing_aiupred:
                raise RuntimeError(msg)
            else:
                if verbose:
                    print("AIUPred not found; skipping disorder feature computation (features will be left missing)")
    
    elif iupred_backend == "iupred2a":
        if not HAS_IUPRED2A:
            raise RuntimeError(
                "IUPred2A backend selected but iupred2a_lib not available.\n"
                "Ensure IUPred2A is installed and iupred2a_lib is importable:\n"
                "  import sys\n"
                "  sys.path.append('/path/to/iupred2a')\n"
                "  from iupred2a import iupred2a_lib"
            )
        if verbose:
            print("Using IUPred2A backend (iupred2a_lib)")

    # Parallel or serial processing
    worker_args = [
        (
            protein_id,
            sequence,
            iupred_backend,
            allow_missing_aiupred,
            force_cpu,
            verbose,
            iupred_dir,
            anchor_dir,
            domain_overlap_dir,
            smart_domain_matches,
            pfam_domain_matches,
            motif_disordered_hmms,
        )
        for protein_id, sequence in sequences
    ]

    if num_workers is None or num_workers < 1:
        num_workers = 1

    if num_workers == 1:
        for args in worker_args:
            _process_protein_task(args)
    else:
        max_workers = min(num_workers, len(worker_args))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(_process_protein_task, worker_args))
