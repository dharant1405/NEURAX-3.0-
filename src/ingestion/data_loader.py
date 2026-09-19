"""
src/ingestion/data_loader.py

Discovers, loads, adapts, and validates all dataset files.
Returns internal DataFrames ready for use by the engines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.ingestion.schema_adapter import SchemaAdapter
from src.ingestion.validator import DataValidator
from src.models.internal_schema import DataQualityReport

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads all dataset tables from a directory, adapts columns to
    internal schema, and validates each table.

    Usage
    -----
    loader = DataLoader(schema_adapter, validator)
    tables, reports = loader.load_directory("data/synthetic")
    """

    def __init__(self, adapter: SchemaAdapter, validator: DataValidator):
        self.adapter = adapter
        self.validator = validator

    # ------------------------------------------------------------------
    # Directory load
    # ------------------------------------------------------------------

    def load_directory(self, directory: str | Path
                       ) -> Tuple[Dict[str, pd.DataFrame],
                                  Dict[str, DataQualityReport]]:
        """
        Scan `directory` for all configured table files,
        load + adapt + validate each one.

        Returns
        -------
        tables : dict of {table_key: DataFrame}
        reports : dict of {table_key: DataQualityReport}
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Data directory not found: {directory}")

        tables: Dict[str, pd.DataFrame] = {}
        reports: Dict[str, DataQualityReport] = {}

        for table_key in self.adapter.configured_tables:
            pattern = self.adapter.get_file_pattern(table_key)
            if not pattern:
                continue

            matches = sorted(directory.glob(pattern))
            if not matches:
                logger.info("No file matching '%s' found for table '%s'.", pattern, table_key)
                continue

            # Use the first match (sorted alphabetically)
            file_path = matches[0]
            logger.info("Loading table '%s' from %s", table_key, file_path)

            try:
                raw_df = self._read_file(file_path)
            except Exception as exc:
                logger.error("Failed to read %s: %s", file_path, exc)
                continue

            # Adapt: translate external columns → internal names
            if table_key == "process_parameters":
                adapted_df = self.adapter.adapt_process_parameters(raw_df)
            else:
                adapted_df = self.adapter.adapt(table_key, raw_df)

            # Validate
            report = self.validator.validate(table_key, adapted_df)
            tables[table_key] = adapted_df
            reports[table_key] = report

            logger.info("Table '%s': %d rows, valid=%s", table_key, len(adapted_df), report.is_valid)

        return tables, reports

    # ------------------------------------------------------------------
    # Single file load
    # ------------------------------------------------------------------

    def load_file(self, file_path: str | Path, table_key: str
                  ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Load, adapt, and validate a single file."""
        file_path = Path(file_path)
        raw_df = self._read_file(file_path)

        if table_key == "process_parameters":
            adapted_df = self.adapter.adapt_process_parameters(raw_df)
        else:
            adapted_df = self.adapter.adapt(table_key, raw_df)

        report = self.validator.validate(table_key, adapted_df)
        return adapted_df, report

    # ------------------------------------------------------------------
    # Pivot process parameters
    # ------------------------------------------------------------------

    @staticmethod
    def pivot_process_parameters(process_df: pd.DataFrame) -> pd.DataFrame:
        """
        If process_df has rows of (unit_id, parameter_name, parameter_value),
        pivot to wide format: one column per parameter.
        If already wide, return as-is.
        """
        if "parameter_name" in process_df.columns and "parameter_value" in process_df.columns:
            index_cols = [c for c in ["unit_id", "batch_id", "station_id", "timestamp"]
                          if c in process_df.columns]
            if not index_cols:
                return process_df
            pivoted = process_df.pivot_table(
                index=index_cols,
                columns="parameter_name",
                values="parameter_value",
                aggfunc="mean",
            ).reset_index()
            pivoted.columns.name = None
            return pivoted
        return process_df

    # ------------------------------------------------------------------
    # File reading
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        elif suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        elif suffix == ".json":
            return pd.read_json(path)
        elif suffix in (".xls", ".xlsx"):
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
