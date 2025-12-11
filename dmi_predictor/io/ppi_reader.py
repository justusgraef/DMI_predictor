"""
Module for reading and parsing protein-protein interaction (PPI) files.

Supports various PPI file formats with flexible delimiter handling.
"""

from pathlib import Path
from typing import List, Tuple, Set, Optional
import pandas as pd


class PPIReader:
    """Read and parse protein-protein interaction files."""

    @staticmethod
    def read_ppi_file(
        filepath: str,
        delimiter: Optional[str] = None,
        skip_header: int = 0,
        prot_a_col: int = 0,
        prot_b_col: int = 1,
    ) -> List[Tuple[str, str]]:
        """
        Read a PPI file and return protein pairs.

        Supports tab-delimited, comma-separated, or space-separated formats.

        Args:
            filepath (str): Path to PPI file
            delimiter (str, optional): Column delimiter. If None, auto-detects from file extension
            skip_header (int): Number of rows to skip (default: 0)
            prot_a_col (int): Column index for protein A (default: 0)
            prot_b_col (int): Column index for protein B (default: 1)

        Returns:
            List[Tuple[str, str]]: List of protein pairs (proteinA, proteinB) sorted alphabetically

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid or insufficient columns
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"PPI file not found: {filepath}")

        # Auto-detect delimiter if not provided
        if delimiter is None:
            if filepath.endswith('.csv'):
                delimiter = ','
            elif filepath.endswith('.tsv'):
                delimiter = '\t'
            else:
                delimiter = '\t'  # Default to tab

        try:
            # Read with pandas for flexible parsing
            df = pd.read_csv(
                filepath,
                sep=delimiter,
                skiprows=skip_header,
                header=None,
                dtype=str,
                comment='#',
                skipinitialspace=True,
            )

            # Validate minimum columns
            max_col_needed = max(prot_a_col, prot_b_col)
            if df.shape[1] <= max_col_needed:
                raise ValueError(
                    f"File has {df.shape[1]} columns but need at least "
                    f"{max_col_needed + 1} columns"
                )

            # Extract protein pairs
            pairs = []
            for _, row in df.iterrows():
                prot_a = row[prot_a_col].strip()
                prot_b = row[prot_b_col].strip()

                if not prot_a or not prot_b:
                    continue  # Skip empty entries

                # Normalize pairs (sorted alphabetically)
                pair = tuple(sorted([prot_a, prot_b]))
                pairs.append(pair)

            if not pairs:
                raise ValueError("No valid protein pairs found in file")

            return pairs

        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing PPI file {filepath}: {e}")
        except Exception as e:
            raise IOError(f"Error reading PPI file {filepath}: {e}")

    @staticmethod
    def read_ppi_file_generic(filepath: str) -> List[Tuple[str, str]]:
        """
        Read PPI file with automatic format detection.

        Attempts to parse the file with common delimiters.

        Args:
            filepath (str): Path to PPI file

        Returns:
            List[Tuple[str, str]]: List of protein pairs
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"PPI file not found: {filepath}")

        delimiters = ['\t', ',', ' ', ';']

        for delimiter in delimiters:
            try:
                return PPIReader.read_ppi_file(filepath, delimiter=delimiter)
            except (ValueError, pd.errors.ParserError):
                continue

        raise ValueError(
            f"Could not parse PPI file {filepath} with any supported delimiter"
        )

    @staticmethod
    def get_unique_proteins(pairs: List[Tuple[str, str]]) -> Set[str]:
        """
        Extract unique protein IDs from PPI pairs.

        Args:
            pairs (List[Tuple[str, str]]): List of protein pairs

        Returns:
            Set[str]: Unique protein IDs
        """
        proteins = set()
        for prot_a, prot_b in pairs:
            proteins.add(prot_a)
            proteins.add(prot_b)
        return proteins

    @staticmethod
    def remove_duplicate_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Remove duplicate protein pairs.

        Args:
            pairs (List[Tuple[str, str]]): List of protein pairs

        Returns:
            List[Tuple[str, str]]: Deduplicated list
        """
        return list(set(pairs))

    @staticmethod
    def validate_ppi_file(filepath: str) -> Tuple[bool, Optional[str]]:
        """
        Validate PPI file format.

        Args:
            filepath (str): Path to PPI file

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            path = Path(filepath)
            if not path.exists():
                return False, f"File not found: {filepath}"

            if path.stat().st_size == 0:
                return False, "File is empty"

            PPIReader.read_ppi_file_generic(filepath)
            return True, None

        except Exception as e:
            return False, str(e)
