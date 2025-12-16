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
