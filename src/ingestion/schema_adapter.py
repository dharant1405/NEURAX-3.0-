"""
src/ingestion/schema_adapter.py

Maps external dataset columns → internal InspectIQ schema.
The YAML mapping (configs/schema_mapping.yaml) defines the translation.
Application logic NEVER reads external column names directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MAPPING = _ROOT / "configs" / "schema_mapping.yaml"


class SchemaAdapter:
    """
    Translates a raw external DataFrame (with organizer column names)
    into a DataFrame that uses InspectIQ's internal field names.

    Usage
    -----
    adapter = SchemaAdapter.from_yaml("configs/schema_mapping.yaml")
    internal_df = adapter.adapt("inspection", raw_df)
    """

    def __init__(self, mapping: Dict[str, Any]):
        self._mapping = mapping
        self._version = mapping.get("version", "unknown")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Optional[str | Path] = None) -> "SchemaAdapter":
        """Load adapter from a YAML mapping file."""
        path = Path(path) if path else _DEFAULT_MAPPING
        if not path.exists():
            raise FileNotFoundError(f"Schema mapping not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            mapping = yaml.safe_load(fh)
        logger.info("Loaded schema mapping v%s from %s", mapping.get("version"), path)
        return cls(mapping)

    # ------------------------------------------------------------------
    # Core translation
    # ------------------------------------------------------------------

    def adapt(self, table_key: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns in `df` according to the mapping for `table_key`.

        Parameters
        ----------
        table_key : str
            One of: inspection, products, process_parameters, production,
                    downtime, economics, stations, batches, defect_catalog
        df : pd.DataFrame
            Raw external DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with internal column names. Unknown columns are
            preserved with a ``_ext_`` prefix so no data is lost.
        """
        if table_key not in self._mapping:
            logger.warning("No mapping found for table '%s'. Returning as-is.", table_key)
            return df.copy()

        section = self._mapping[table_key]
        rename_map = self._build_rename_map(section)

        # Separate known and unknown columns
        known_external = set(rename_map.keys())
        unknown_cols = [c for c in df.columns if c not in known_external]

        adapted = df.rename(columns=rename_map)

        # Preserve unknown columns with prefix
        if unknown_cols:
            logger.debug("Table '%s': %d unknown columns kept with '_ext_' prefix: %s",
                         table_key, len(unknown_cols), unknown_cols[:5])
            adapted = adapted.rename(columns={c: f"_ext_{c}" for c in unknown_cols
                                              if c in adapted.columns})

        return adapted

    def reverse_adapt(self, table_key: str, df: pd.DataFrame) -> pd.DataFrame:
        """Translate internal names back to external names (for export)."""
        if table_key not in self._mapping:
            return df.copy()
        section = self._mapping[table_key]
        rename_map = {v: k for k, v in self._build_rename_map(section).items()}
        return df.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # Process parameter table: has nested structure
    # ------------------------------------------------------------------

    def adapt_process_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapt the process parameters table.
        The parameters sub-section maps parameter names dynamically.
        """
        section = self._mapping.get("process_parameters", {})
        top_rename = {}
        for internal_key, ext_col in section.items():
            if internal_key in ("file_pattern", "parameters") or ext_col is None:
                continue
            top_rename[ext_col] = internal_key

        param_section = section.get("parameters", {})
        param_rename = {}
        for internal_param, ext_col in param_section.items():
            if ext_col and ext_col in df.columns:
                param_rename[ext_col] = internal_param

        rename_map = {**top_rename, **param_rename}
        return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ------------------------------------------------------------------
    # Column availability checks
    # ------------------------------------------------------------------

    def available_internal_fields(self, table_key: str, df: pd.DataFrame) -> List[str]:
        """Return internal field names that exist in the adapted DataFrame."""
        adapted = self.adapt(table_key, df)
        section = self._mapping.get(table_key, {})
        internal_fields = [v for k, v in self._build_rename_map(section).items()]
        return [f for f in internal_fields if f in adapted.columns]

    def missing_fields(self, table_key: str, df: pd.DataFrame,
                       required: List[str]) -> List[str]:
        """Return required internal fields missing from the adapted DataFrame."""
        available = self.available_internal_fields(table_key, df)
        return [f for f in required if f not in available]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_rename_map(self, section: Dict[str, Any]) -> Dict[str, str]:
        """
        Build external→internal rename dict from a mapping section.
        Skips null values and meta-keys (file_pattern, parameters).
        """
        skip_keys = {"file_pattern", "parameters", "version"}
        rename_map: Dict[str, str] = {}
        for internal_key, ext_col in section.items():
            if internal_key in skip_keys:
                continue
            if ext_col is None:
                continue
            if isinstance(ext_col, str):
                rename_map[ext_col] = internal_key
        return rename_map

    def get_file_pattern(self, table_key: str) -> Optional[str]:
        """Get the file glob pattern for a table."""
        section = self._mapping.get(table_key, {})
        return section.get("file_pattern")

    @property
    def version(self) -> str:
        return self._version

    @property
    def configured_tables(self) -> List[str]:
        meta_keys = {"version"}
        return [k for k in self._mapping if k not in meta_keys]
