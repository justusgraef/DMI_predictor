"""
Configuration management for DMI Predictor.

Handles data paths, model loading, and parameter configuration.
"""

from pathlib import Path
from typing import Optional
import json
import os


class DMIPredictorConfig:
    """Configuration for DMI Predictor."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize DMI Predictor configuration.

        Args:
            data_dir (str, optional): Path to data directory. If None, uses package default.
        """
        self.package_dir = Path(__file__).parent.parent
        self.data_dir = Path(data_dir) if data_dir else self.package_dir / "data"
        self.scripts_dir = self.package_dir.parent / "scripts"

        # Data files
        self.elm_classes_file = self.data_dir / "20220311_elm_classes.tsv"
        self.elm_dmi_file = self.data_dir / "20220311_elm_interaction_domains_complete.tsv"
        self.pfam_freq_file = self.data_dir / "all_pfam_domains_with_frequency.txt"
        self.smart_freq_file = self.data_dir / "all_smart_domains_with_frequency.txt"
        self.training_data_file = self.data_dir / "final_PRS_RRSv4_3_used_to_fit_model.tsv"
        self.model_file = self.data_dir / "final_RF_model_with_RRSv4_3.joblib"
        self.imputer_file = self.data_dir / "final_median_imputer_with_RRSv4_3.joblib"
        self.interpro_pfam_file = self.data_dir / "interpro_9606_pfam_matches_20210122.json"
        self.interpro_smart_file = self.data_dir / "interpro_9606_smart_matches_20210122.json"

        # Note: do not validate files on import/initialization to allow importing
        # the package in environments where the large data files live elsewhere.
        # Call `validate()` explicitly where needed.

    def _validate_data_files(self) -> None:
        """
        Validate that required data files exist.

        Raises:
            FileNotFoundError: If critical data files are missing
        """
        critical_files = [
            self.model_file,
            self.imputer_file,
            self.elm_classes_file,
            self.elm_dmi_file,
        ]

        missing_files = [f for f in critical_files if not f.exists()]

        if missing_files:
            raise FileNotFoundError(
                f"Missing critical data files:\n"
                f"{chr(10).join(str(f) for f in missing_files)}\n"
                f"Please ensure all data files are in: {self.data_dir}"
            )

    def get_data_file_status(self) -> dict:
        """
        Get status of all data files.

        Returns:
            dict: Dictionary with file paths and existence status
        """
        files = {
            "elm_classes": self.elm_classes_file,
            "elm_dmi": self.elm_dmi_file,
            "pfam_freq": self.pfam_freq_file,
            "smart_freq": self.smart_freq_file,
            "training_data": self.training_data_file,
            "model": self.model_file,
            "imputer": self.imputer_file,
            "interpro_pfam": self.interpro_pfam_file,
            "interpro_smart": self.interpro_smart_file,
        }

        status = {}
        for name, filepath in files.items():
            status[name] = {
                "path": str(filepath),
                "exists": filepath.exists(),
                "size_mb": (
                    filepath.stat().st_size / (1024 * 1024)
                    if filepath.exists()
                    else None
                ),
            }

        return status

    def to_dict(self) -> dict:
        """
        Export configuration as dictionary.

        Returns:
            dict: Configuration dictionary
        """
        return {
            "data_dir": str(self.data_dir),
            "package_dir": str(self.package_dir),
            "elm_classes_file": str(self.elm_classes_file),
            "elm_dmi_file": str(self.elm_dmi_file),
            "pfam_freq_file": str(self.pfam_freq_file),
            "smart_freq_file": str(self.smart_freq_file),
            "model_file": str(self.model_file),
            "imputer_file": str(self.imputer_file),
            "interpro_pfam_file": str(self.interpro_pfam_file),
            "interpro_smart_file": str(self.interpro_smart_file),
        }
