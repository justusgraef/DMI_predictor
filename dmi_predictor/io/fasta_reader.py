"""
Module for reading and parsing FASTA files.

This module provides utilities to read protein sequences from FASTA files,
supporting both single and batch FASTA file operations.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import re


class FastaReader:
    """Read and parse FASTA format files."""

    @staticmethod
    def read_fasta_file(filepath: str) -> Dict[str, str]:
        """
        Read a FASTA file and return protein sequences.

        Args:
            filepath (str): Path to FASTA file

        Returns:
            Dict[str, str]: Dictionary with protein IDs as keys and sequences as values

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If FASTA format is invalid
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"FASTA file not found: {filepath}")

        sequences = {}
        current_id = None
        current_seq = []

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('>'):
                        # Save previous sequence if exists
                        if current_id is not None:
                            sequences[current_id] = ''.join(current_seq)

                        # Parse header - extract first token as ID
                        header = line[1:]  # Remove '>'
                        current_id = header.split()[0] if header.split() else header
                        current_seq = []

                        if not current_id:
                            raise ValueError("Empty protein ID in FASTA header")
                    else:
                        # Validate sequence contains only amino acids
                        if not re.match(r'^[A-Z*\-]+$', line, re.IGNORECASE):
                            raise ValueError(
                                f"Invalid characters in sequence for {current_id}: {line}"
                            )
                        current_seq.append(line.upper())

                # Save last sequence
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)

            if not sequences:
                raise ValueError("No sequences found in FASTA file")

            return sequences

        except IOError as e:
            raise IOError(f"Error reading FASTA file {filepath}: {e}")

    @staticmethod
    def read_fasta_directory(dirpath: str, pattern: str = "*.fasta") -> Dict[str, str]:
        """
        Read all FASTA files from a directory.

        Args:
            dirpath (str): Path to directory containing FASTA files
            pattern (str): Glob pattern for FASTA files (default: "*.fasta")

        Returns:
            Dict[str, str]: Combined dictionary of protein IDs and sequences

        Raises:
            FileNotFoundError: If directory does not exist or no FASTA files found
        """
        dir_path = Path(dirpath)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")

        all_sequences = {}
        fasta_files = list(dir_path.glob(pattern))

        if not fasta_files:
            raise FileNotFoundError(
                f"No FASTA files matching pattern '{pattern}' found in {dirpath}"
            )

        for fasta_file in fasta_files:
            try:
                sequences = FastaReader.read_fasta_file(str(fasta_file))
                all_sequences.update(sequences)
            except (FileNotFoundError, ValueError, IOError) as e:
                raise IOError(f"Error reading {fasta_file.name}: {e}")

        return all_sequences

    @staticmethod
    def validate_fasta(filepath: str) -> Tuple[bool, Optional[str]]:
        """
        Validate FASTA file format.

        Args:
            filepath (str): Path to FASTA file

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            path = Path(filepath)
            if not path.exists():
                return False, f"File not found: {filepath}"

            with open(filepath, 'r') as f:
                first_char = f.read(1)
                if not first_char:
                    return False, "File is empty"
                if first_char != '>':
                    return False, "File does not start with '>' (not valid FASTA)"

            # Try to read it
            FastaReader.read_fasta_file(filepath)
            return True, None

        except Exception as e:
            return False, str(e)
