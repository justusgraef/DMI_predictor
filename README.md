# DMI Predictor

Predict **Domain-Motif Interfaces (DMI)** in protein-protein interactions using machine learning.

## Overview

DMI Predictor automates the prediction of functional domain-motif interactions in protein sequences. It:

1. **Detects domains** in protein sequences using InterPro/Pfam/SMART databases
2. **Identifies short linear motifs (SLiMs)** using ELM database patterns
3. **Matches domains to motifs** using a curated reference database
4. **Scores predictions** using a trained random forest model
5. **Ranks results** by likelihood of functional interaction

## Quick Start

### Installation

```bash
git clone https://github.com/lagelab/DMI_predictor.git
cd DMI_predictor
pip install -e .

# Required for precompute: AIUPred disorder scores
# For CPU-only:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   git clone https://github.com/doszilab/AIUPred.git /tmp/AIUPred
#   export PYTHONPATH=/tmp/AIUPred:$PYTHONPATH
```

### Basic Usage

```bash
# Run prediction on PPI list with sequences
dmi-predict predict \
  --ppi-file interactions.tsv \
  --fasta-dir ./sequences/ \
  --output results.tsv

# Precompute disorder features (folder of FASTA files)
dmi-predict precompute-features \
  --fasta-dir ./sequences/ \
  --output-dir ./features \
  --verbose

# Or from a single multi-FASTA
dmi-predict precompute-features \
  --fasta-file ./proteins.fasta \
  --output-dir ./features \
  --verbose
```

For detailed usage, see [USAGE.md](USAGE.md)

## Features

✨ **Easy-to-use CLI**
- Simple commands for common workflows
- Flexible input formats (TSV, CSV, FASTA)
- Multiple output formats (TSV, CSV, JSON)

⚡ **Scalable**
- Batch processing of PPIs
- Flexible sequence input (directory or individual files)
- Configurable scoring thresholds

🧬 **Comprehensive**
- 1000+ ELM motif patterns
- Pfam and SMART domain databases
- 16+ prediction features per DMI

🔬 **Well-validated**
- Trained on curated DMI datasets
- Median imputation for missing features
- Cross-validated random forest model

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

scripts/             # Research/development scripts
├── DMI_prediction/  # Core prediction algorithms
├── RRS_formation/   # Random reference set generation
├── features_analysis/
└── model_fitting_evaluation/
```

### Data Requirements

Critical data files (place in `dmi_predictor/data/`):
- `20220311_elm_classes.tsv` - ELM motif definitions
- `20220311_elm_interaction_domains_complete.tsv` - DMI type database
- `final_RF_model_with_RRSv4_3.joblib` - Trained random forest model
- `final_median_imputer_with_RRSv4_3.joblib` - Feature imputer
- `interpro_9606_pfam_matches_20210122.json` - InterPro domain matches
- `interpro_9606_smart_matches_20210122.json` - InterPro domain matches

## Usage Examples

### Command Line

```bash
# Validate data files
dmi-predict check-data

# Validate input files
dmi-predict validate-ppi --ppi-file interactions.tsv
dmi-predict validate-fasta --fasta-file proteins.fasta

# Run prediction with custom threshold
dmi-predict predict \
  --ppi-file interactions.tsv \
  --fasta-dir sequences/ \
  --score-threshold 0.6 \
  --output-format json \
  --output results.json \
  --verbose
```

### Python API

```python
from dmi_predictor import DMIPredictorConfig, FastaReader, PPIReader

# Load configuration
config = DMIPredictorConfig()

# Read PPI file
ppi_pairs = PPIReader.read_ppi_file_generic('interactions.tsv')

# Read sequences
sequences = FastaReader.read_fasta_directory('./sequences/')

# Run prediction (coming soon)
# results = predictor.predict(ppi_pairs, sequences, config)
```

See [examples/example_workflow.py](examples/example_workflow.py) for complete example.

## Development

Original research and development scripts are in `scripts/` directory:

- **DMI_prediction/**: Core prediction algorithms (DMIDB.py, DMIpredictor.py)
- **RRS_formation/**: Random reference set generation
- **features_analysis/**: Feature importance and model evaluation
- **model_fitting_evaluation/**: Model training and cross-validation

These scripts are preserved for:
- Model retraining with new data
- Feature analysis and visualization
- Validation and benchmarking
- Reference for understanding algorithms

## Troubleshooting

**Missing data files?**
```bash
dmi-predict check-data  # Check which files are missing
```

**Invalid FASTA file?**
```bash
dmi-predict validate-fasta --fasta-file proteins.fasta --verbose
```

**Invalid PPI file?**
```bash
dmi-predict validate-ppi --ppi-file interactions.tsv --verbose
```

See [USAGE.md](USAGE.md) for comprehensive troubleshooting guide.

## Requirements

- Python 3.8+
- scikit-learn >= 0.23.0
- numpy >= 1.19.0
- pandas >= 1.1.0
- click >= 8.0.0

See `requirements.txt` for full list.

## Citation

Original work:
```
[Original citation - add reference to Chop Yan Lee et al.]
```

## License

MIT License - see [LICENSE](LICENSE) file

## Authors

- Chop Yan Lee (Original developer)
- Justus Graef (Original developer)
- [Additional contributors]