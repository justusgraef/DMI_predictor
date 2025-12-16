"""
Command-line interface for DMI Predictor.

Provides intuitive CLI for running DMI predictions on PPI pairs with protein sequences.
"""

import click
import sys
from pathlib import Path
from typing import Optional

from dmi_predictor import DMIPredictorConfig, FastaReader, PPIReader
from dmi_predictor.io.result_writer import ResultWriter


@click.group()
@click.version_option()
def cli():
    """DMI Predictor - Predict domain-motif interfaces in protein-protein interactions."""
    pass


@cli.command()
@click.option(
    '--ppi-file',
    type=click.Path(exists=True),
    required=True,
    help='Path to PPI file (TSV/CSV with two columns: ProteinA ProteinB)',
)
@click.option(
    '--fasta-dir',
    type=click.Path(exists=True),
    required=False,
    help='Path to directory containing FASTA files (one per protein or combined)',
)
@click.option(
    '--fasta-files',
    type=click.Path(exists=True),
    multiple=True,
    required=False,
    help='Individual FASTA files (can be specified multiple times)',
)
@click.option(
    '--output',
    type=click.Path(),
    default='dmi_predictions.tsv',
    help='Output file for predictions (default: dmi_predictions.tsv)',
)
@click.option(
    '--output-format',
    type=click.Choice(['tsv', 'csv', 'json']),
    default='tsv',
    help='Output format (default: tsv)',
)
@click.option(
    '--score-threshold',
    type=float,
    default=0.5,
    help='Score threshold for reporting DMIs (default: 0.5)',
)
@click.option(
    '--data-dir',
    type=click.Path(exists=True),
    required=False,
    help='Path to data directory (uses package default if not provided)',
)
@click.option(
    '--features-dir',
    type=click.Path(exists=True),
    required=False,
    help='Path to precomputed feature directory (IUPred_short/Anchor/Domain_overlap)',
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Enable verbose output',
)
@click.option(
    '--num-workers',
    type=int,
    default=1,
    help='Parallel workers for prediction. Default: 1 (serial).',
)
@click.option(
    '--skip-elm',
    is_flag=True,
    help='Skip ELM network calls (use cached/none defined-positions). Useful for offline runs.',
)
def predict(
    ppi_file: str,
    fasta_dir: Optional[str],
    fasta_files: tuple,
    output: str,
    output_format: str,
    score_threshold: float,
    data_dir: Optional[str],
    features_dir: Optional[str],
    verbose: bool,
    num_workers: int,
    skip_elm: bool,
):
    """
    Predict domain-motif interfaces (DMI) for protein-protein interactions.

    Example usage:

        dmi-predict predict --ppi-file interactions.tsv --fasta-dir ./sequences/ --output results.tsv

        dmi-predict predict --ppi-file interactions.tsv --fasta-files prot1.fasta prot2.fasta --output results.json --output-format json
    """
    try:
        if verbose:
            click.echo("DMI Predictor - Starting prediction pipeline")
            click.echo(f"PPI file: {ppi_file}")

        # Initialize configuration
        config = DMIPredictorConfig(data_dir=data_dir)
        if verbose:
            click.echo("Configuration loaded successfully")

        # Read PPI file
        if verbose:
            click.echo(f"Reading PPI file: {ppi_file}")
        ppi_pairs = PPIReader.read_ppi_file_generic(ppi_file)
        unique_proteins = PPIReader.get_unique_proteins(ppi_pairs)
        if verbose:
            click.echo(
                f"Found {len(ppi_pairs)} protein pairs with {len(unique_proteins)} unique proteins"
            )

        # Read sequences
        if verbose:
            click.echo("Reading protein sequences...")
        sequences = {}

        if fasta_dir:
            fasta_sequences = FastaReader.read_fasta_directory(fasta_dir)
            sequences.update(fasta_sequences)
            if verbose:
                click.echo(f"Loaded {len(fasta_sequences)} sequences from {fasta_dir}")

        if fasta_files:
            for fasta_file in fasta_files:
                fasta_sequences = FastaReader.read_fasta_file(fasta_file)
                sequences.update(fasta_sequences)
                if verbose:
                    click.echo(f"Loaded {len(fasta_sequences)} sequences from {fasta_file}")

        if not sequences:
            raise ValueError("No sequences provided. Use --fasta-dir or --fasta-files")

        # Validate all proteins have sequences
        missing_proteins = unique_proteins - set(sequences.keys())
        if missing_proteins:
            click.echo(
                click.style(
                    f"⚠ Warning: {len(missing_proteins)} proteins have no sequences: "
                    f"{', '.join(sorted(missing_proteins)[:5])}{'...' if len(missing_proteins) > 5 else ''}",
                    fg='yellow',
                ),
                err=True,
            )

        if verbose:
            click.echo(
                f"Found sequences for {len(sequences)} proteins "
                f"({len(unique_proteins - set(sequences.keys()))} missing)"
            )

        # Run prediction pipeline (core workflow)
        from dmi_predictor.core.predictor import run_dmi_prediction
        run_dmi_prediction(
            ppi_pairs=ppi_pairs,
            sequences=sequences,
            config=config,
            output_file=output,
            output_format=output_format,
            score_threshold=score_threshold,
            verbose=verbose,
            features_dir=features_dir,
            skip_elm=skip_elm,
            num_workers=num_workers,
        )

        if verbose:
            click.echo(f"Configuration: {config.to_dict()}")

    except FileNotFoundError as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'), err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"✗ Unexpected error: {e}", fg='red'), err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    '--data-dir',
    type=click.Path(exists=True),
    required=False,
    help='Path to data directory',
)
def check_data(data_dir: Optional[str]):
    """Check status of required data files."""
    try:
        config = DMIPredictorConfig(data_dir=data_dir)
        status = config.get_data_file_status()

        click.echo("\nData File Status:")
        click.echo("-" * 80)

        all_present = True
        for name, info in status.items():
            exists = info['exists']
            size = info.get('size_mb')
            status_text = click.style("✓", fg='green') if exists else click.style("✗", fg='red')

            size_str = f" ({size:.1f} MB)" if size else " (missing)"
            click.echo(f"{status_text} {name:20s} {size_str}")
            all_present = all_present and exists

        click.echo("-" * 80)
        if all_present:
            click.echo(click.style("✓ All data files present!", fg='green'))
        else:
            click.echo(click.style("✗ Some data files are missing", fg='red'), err=True)
            click.echo("\nTo set up data files:")
            click.echo("1. Download InterPro JSON files from InterPro")
            click.echo("2. Place all data files in the data directory:")
            click.echo(f"   {config.data_dir}")

    except FileNotFoundError as e:
        click.echo(click.style(f"✗ Error: {e}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--ppi-file',
    type=click.Path(exists=True),
    required=True,
    help='Path to PPI file to validate',
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed output',
)
def validate_ppi(ppi_file: str, verbose: bool):
    """Validate PPI file format."""
    is_valid, error_msg = PPIReader.validate_ppi_file(ppi_file)

    if is_valid:
        pairs = PPIReader.read_ppi_file_generic(ppi_file)
        click.echo(click.style(f"✓ Valid PPI file with {len(pairs)} pairs", fg='green'))

        if verbose:
            unique_proteins = PPIReader.get_unique_proteins(pairs)
            click.echo(f"  Unique proteins: {len(unique_proteins)}")
            click.echo(f"  Sample pairs: {pairs[:3]}")
    else:
        click.echo(click.style(f"✗ Invalid PPI file: {error_msg}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--fasta-file',
    type=click.Path(exists=True),
    required=True,
    help='Path to FASTA file to validate',
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Show detailed output',
)
def validate_fasta(fasta_file: str, verbose: bool):
    """Validate FASTA file format."""
    is_valid, error_msg = FastaReader.validate_fasta(fasta_file)

    if is_valid:
        sequences = FastaReader.read_fasta_file(fasta_file)
        total_aa = sum(len(seq) for seq in sequences.values())
        click.echo(
            click.style(
                f"✓ Valid FASTA file with {len(sequences)} sequences ({total_aa} amino acids)",
                fg='green',
            )
        )

        if verbose:
            for prot_id, seq in list(sequences.items())[:3]:
                click.echo(f"  {prot_id}: {len(seq)} aa")
            if len(sequences) > 3:
                click.echo(f"  ... and {len(sequences) - 3} more")
    else:
        click.echo(click.style(f"✗ Invalid FASTA file: {error_msg}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--fasta-dir',
    type=click.Path(exists=True),
    required=False,
    help='Directory with FASTA files (one per protein)',
)
@click.option(
    '--fasta-file',
    type=click.Path(exists=True),
    required=False,
    help='Single FASTA file (multi-FASTA supported)',
)
@click.option(
    '--output-dir',
    type=click.Path(),
    required=True,
    help='Directory to write precomputed features',
)
@click.option(
    '--force-cpu',
    is_flag=True,
    help='Force AIUPred to use CPU instead of GPU',
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Enable verbose output',
)
@click.option(
    '--allow-missing-aiupred',
    is_flag=True,
    help='If set, skip AIUPred when it is not installed and leave disorder features missing (they will be imputed downstream)',
)
@click.option(
    '--write-placeholders',
    is_flag=True,
    help='If set, write placeholder feature files when features cannot be computed (not recommended)',
)
@click.option(
    '--iupred-backend',
    type=click.Choice(['aiupred', 'iupred2a']),
    default='aiupred',
    help='Which IUPred backend to use for disorder prediction (default: aiupred). Requires iupred2a_lib to be importable for iupred2a backend.',
)
@click.option(
    '--num-workers',
    type=int,
    default=1,
    help='Parallel workers for feature precompute (per-protein). Default: 1 (serial).',
)
def precompute_features(fasta_dir, fasta_file, output_dir, force_cpu, verbose, allow_missing_aiupred, write_placeholders, iupred_backend, num_workers):
    """
    Precompute per-protein features (AIUPred disorder/Anchor, DomainOverlap).
    Accepts a directory of FASTA files or a single multi-FASTA.
    """
    if bool(fasta_dir) == bool(fasta_file):
        click.echo(
            click.style("✗ Provide exactly one of --fasta-dir or --fasta-file", fg='red'),
            err=True,
        )
        sys.exit(1)
    from dmi_predictor.core.precompute import precompute_features as workflow_precompute
    try:
        # Pass through new flags (allow skipping AIUPred and optional placeholders)
        workflow_precompute(
            fasta_dir=fasta_dir,
            fasta_file=fasta_file,
            output_dir=output_dir,
            verbose=verbose,
            force_cpu=force_cpu,
            allow_missing_aiupred=allow_missing_aiupred,
            write_placeholders=write_placeholders,
            iupred_backend=iupred_backend,
            num_workers=num_workers,
        )
    except RuntimeError as e:
        click.echo(click.style(f"✗ {e}", fg='red'), err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
