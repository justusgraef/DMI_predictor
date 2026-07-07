# DMI Predictor

Predict **Domain-Motif Interfaces (DMI)** in protein-protein interactions using machine learning.

## Overview

DMI Predictor automates the prediction of functional domain-motif interactions in protein sequences. It:

1. **Detects domains** in protein sequences using InterPro/Pfam/SMART databases
2. **Identifies short linear motifs (SLiMs)** using ELM database patterns
3. **Matches domains to motifs** using a curated reference database
4. **Scores predictions** using a pre-trained random forest model
5. **Ranks results** by likelihood of functional interaction

## Quick Start

### Installation

```bash
git clone https://github.com/justusgraef/DMI_predictor.git
cd DMI_predictor
pip install -e .

```

#### Conda environment (Python 3.9 required)

`scikit-learn==0.24.1` only ships wheels for Python 3.8/3.9; use Python 3.9 to avoid source builds failing on newer Pythons.
`scikit-learn==0.24.1` is needed as it was used during model pre-training.

```bash

# create and activate the environment with Python 3.9
conda create -y -n dmi-env python=3.9
conda activate dmi-env

# verify interpreter is from the env and 3.9.x
which python
python --version

# install the package and deps inside the env
pip install -e .
```

#### IUPred2A / AIUPred

- Preferred: install IUPred2A (includes ANCHOR2) and make its Python package importable, e.g. `export PYTHONPATH=/path/to/iupred2a:$PYTHONPATH`. Download: https://iupred2a.elte.hu/download_new
- Alternative: AIUPred backend (`--iupred-backend aiupred`) if IUPred2A is unavailable. Download: https://aiupred.elte.hu/download

### Minimal usage

Precompute features (disorder, ANCHOR2, domain overlap) from a multi-FASTA using IUPred2A:

```bash
export PYTHONPATH=/path/to/iupred2a:$PYTHONPATH
dmi-predict precompute-features \
  --fasta-file sequences.fasta \
  --output-dir features \
  --iupred-backend iupred2a \
  --verbose
```

Run prediction on a PPI table with the corresponding sequences and precomputed features:

```bash
dmi-predict predict \
  --ppi-file interactions.tsv \
  --fasta-files sequences.fasta \
  --features-dir features \
  --output results.tsv \
  --verbose
```

### Domain position correction

The DMI predictor uses bundled InterPro domain annotations (January 2021). For proteins whose canonical sequence length has changed since then, the domain boundary positions in the output may be mismatched. Use `--update-interpro-domains` on the `predict` command to query the current InterPro REST API and overwrite `DomainMatch1` / `DomainMatch2` with up-to-date coordinates. Original positions are preserved in `DomainMatch1_DMI` / `DomainMatch2_DMI`.

```bash
dmi-predict predict \
  --ppi-file interactions.tsv \
  --fasta-files sequences.fasta \
  --features-dir features \
  --output results.tsv \
  --update-interpro-domains \
  --verbose
```

A JSON cache file (`interpro_cache.json` by default, alongside the output) stores API results so repeated runs do not re-query the same proteins. A pre-existing cache can be supplied with `--interpro-cache /path/to/cache.json`. The API request rate can be controlled with `--interpro-api-delay` (default: 0.25 s).

### Conservation score re-alignment

Conservation scores (`conservation_scores/<UniProt>_con.json`) are stored keyed by **absolute residue position**, computed against the sequences that were canonical when the score library was built. If a protein's UniProt sequence is later updated (e.g. a new isoform that adds N-terminal residues), position *N* no longer refers to the same residue, so `predict` silently reads misaligned conservation and mis-scores that protein's DMIs — the conservation features (`qfo_RLC`, `vertebrates_RLC`, …) feed the random-forest score, so a shift of even a few residues can push real hits below the cutoff.

**When to use:** whenever your input sequences differ from the ones the conservation library was built against (a sequence "data freeze" update). This is the conservation-score counterpart of `--update-interpro-domains`; both re-align position-indexed precomputed data to the current sequences.

Use `--remap-conservation` on the `predict` command, together with `--conservation-ref-fasta` pointing at the FASTA the library was built against (the "old" sequences):

```bash
dmi-predict predict \
  --ppi-file interactions.tsv \
  --fasta-files sequences.fasta \
  --features-dir features \
  --conservation-scores-dir /path/to/conservation_scores \
  --output results.tsv \
  --remap-conservation \
  --conservation-ref-fasta old_sequences.fasta \
  --verbose
```

For each protein whose sequence changed, existing per-residue scores are relocated to their new coordinates via an old→new sequence alignment (handles N-/C-terminal extensions, trims, and internal indels). Corrected files are written into `<features-dir>/conservation_scores/`, which the loader prefers over the shared library **per protein** — so only re-aligned proteins are overridden and the source library is left untouched. `--features-dir` is therefore required with this flag.

**Note:** this re-uses existing scores; it does *not* regenerate conservation from orthologs. Residues that exist only in the new sequence (e.g. a brand-new N-terminus) are left unscored and fall back to median imputation, exactly as any protein lacking conservation data. To obtain true conservation for genuinely new residues, rerun the original conservation pipeline.

### Variant mapping

Map curated protein variants (from the EBI Proteins API) onto predicted DMI interface windows to identify variants overlapping SLiM or domain positions:

```bash
dmi-predict map-variants \
  --input results.tsv \
  --output mapped_variants.csv \
  --dmi-score-cutoff 0.7 \
  --verbose
```

`--input` is the TSV output from `dmi-predict predict`. `--dmi-score-cutoff` pre-filters to high-confidence predictions before querying the API. Output is a CSV with one row per variant–window overlap, carrying all DMI result columns plus variant annotation columns (`position`, `wild_type`, `mutant`, `consequence_type`, `clinical_significance`, `xrefs`). A `window_type` column indicates whether the variant falls in the `SLiM` or `Domain` window of the predicted interface.

### Parallel processing (HPC / multi-core)

Both `precompute-features` and `predict` support `--num-workers` for parallel execution:

```bash
# Precompute with 32 workers (per-protein parallelization)
dmi-predict precompute-features \
  --fasta-file sequences.fasta \
  --output-dir features \
  --iupred-backend iupred2a \
  --num-workers 32 \
  --verbose

# Predict with 32 workers (per-PPI-chunk parallelization)
dmi-predict predict \
  --ppi-file interactions.tsv \
  --fasta-files sequences.fasta \
  --features-dir features \
  --output results.tsv \
  --num-workers 32 \
  --verbose
```

Speedup scales nearly linearly with cores for large datasets. Default is `--num-workers 1` (serial).

### Input formats

- **PPI file (`--ppi-file`)**: TSV/CSV with at least two columns: protein A, protein B. No header preferred; comment lines starting with `#` are ignored. Delimiter auto-detected from extension (`.tsv` -> tab, `.csv` -> comma, otherwise tab). Protein IDs should match the FASTA headers (first token).
- **Sequences**: Either a single multi-FASTA (`--fasta-files sequences.fasta`) or a directory of FASTA files (`--fasta-dir ./sequences/`). For each record, the first whitespace-separated token after `>` is taken as the ID. Sequences must use amino-acid letters A–Z (and `*`/`-` if present). IDs must match those used in the PPI file.




## Architecture

### Package Structure

```
dmi_predictor/
├── cli/              # Command-line interface
├── io/               # Input/output utilities
│   ├── fasta_reader.py
│   ├── ppi_reader.py
│   └── result_writer.py
├── core/             # Core prediction logic
├── config.py         # Configuration management
└── data/             # Data files (downloaded separately)

```

### Data Requirements

Critical data files (place in `dmi_predictor/data/`):
- `20220311_elm_classes.tsv` - ELM motif definitions
- `20220311_elm_interaction_domains_complete.tsv` - DMI type database
- `final_RF_model_with_RRSv4_3.joblib` - Trained random forest model
- `final_median_imputer_with_RRSv4_3.joblib` - Feature imputer
- `interpro_9606_pfam_matches_20210122.json` - InterPro domain matches
- `interpro_9606_smart_matches_20210122.json` - InterPro domain matches
- `motif_disordered_smart_pfam_hmms.json` - HMMs to exclude from domain-overlap (Pfam "Disordered", SMART "Motif")


## Citation

Original work:
```
[Hubrich, Valverde, Lee et al., 2025, bioRxiv, doi: 10.1101/2025.06.27.661911]
```

## License

MIT License - see [LICENSE](LICENSE) file
