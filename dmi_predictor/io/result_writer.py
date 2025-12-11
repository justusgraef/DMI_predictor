"""
Module for writing DMI prediction results in various formats.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import csv


class ResultWriter:
    """Write DMI prediction results to various formats."""

    @staticmethod
    def write_tsv(
        results: List[Dict[str, Any]],
        output_file: str,
        fieldnames: Optional[List[str]] = None,
    ) -> None:
        """
        Write results to TSV file.

        Args:
            results (List[Dict]): List of result dictionaries
            output_file (str): Path to output file
            fieldnames (List[str], optional): Column names. If None, uses first result keys.
        """
        if not results:
            raise ValueError("No results to write")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fieldnames is None:
            fieldnames = list(results[0].keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(results)

    @staticmethod
    def write_csv(
        results: List[Dict[str, Any]],
        output_file: str,
        fieldnames: Optional[List[str]] = None,
    ) -> None:
        """
        Write results to CSV file.

        Args:
            results (List[Dict]): List of result dictionaries
            output_file (str): Path to output file
            fieldnames (List[str], optional): Column names. If None, uses first result keys.
        """
        if not results:
            raise ValueError("No results to write")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fieldnames is None:
            fieldnames = list(results[0].keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    @staticmethod
    def write_json(
        results: List[Dict[str, Any]],
        output_file: str,
        pretty: bool = True,
    ) -> None:
        """
        Write results to JSON file.

        Args:
            results (List[Dict]): List of result dictionaries
            output_file (str): Path to output file
            pretty (bool): Pretty print JSON (default: True)
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            if pretty:
                json.dump(results, f, indent=2)
            else:
                json.dump(results, f)

    @staticmethod
    def write_summary(
        summary: Dict[str, Any],
        output_file: str,
    ) -> None:
        """
        Write summary statistics to JSON file.

        Args:
            summary (Dict): Summary dictionary
            output_file (str): Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
