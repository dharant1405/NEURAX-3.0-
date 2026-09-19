"""
src/ingestion/validator.py

Validates an adapted DataFrame against the internal schema.
Produces a DataQualityReport without modifying the data.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from src.models.internal_schema import DataQualityReport

logger = logging.getLogger(__name__)


# Required internal fields per table
# Long-format required fields
REQUIRED_FIELDS: Dict[str, List[str]] = {
    "inspection": ["unit_id"],
    "products":   ["unit_id"],
    # process_parameters: can be wide-format (unit_id + parameter columns)
    # OR long-format (station_id + parameter_name + parameter_value)
    # Validation is permissive — just needs station_id OR unit_id
    "process_parameters": ["station_id"],
    "production": ["batch_id", "station_id"],
    "downtime":   ["station_id", "duration_minutes"],
    "economics":  [],
    "stations":   ["station_id"],
    "batches":    ["batch_id"],
    "defect_catalog": ["defect_type", "defect_family"],
}


class DataValidator:
    """
    Validates adapted DataFrames.

    Usage
    -----
    validator = DataValidator()
    report = validator.validate("inspection", df)
    """

    def __init__(self, null_warn_threshold: float = 0.30,
                 null_error_threshold: float = 0.80):
        """
        Parameters
        ----------
        null_warn_threshold : float
            Fraction of nulls in a column that triggers a WARNING.
        null_error_threshold : float
            Fraction of nulls in a column that triggers an ERROR.
        """
        self.null_warn_threshold = null_warn_threshold
        self.null_error_threshold = null_error_threshold

    # ------------------------------------------------------------------
    # Main validate entry point
    # ------------------------------------------------------------------

    def validate(self, table_key: str, df: pd.DataFrame,
                 extra_required: Optional[List[str]] = None) -> DataQualityReport:
        """
        Run all validation checks and return a DataQualityReport.

        Parameters
        ----------
        table_key : str
            Table identifier (must be in REQUIRED_FIELDS or 'unknown').
        df : pd.DataFrame
            Adapted DataFrame with internal field names.
        extra_required : list, optional
            Additional required fields for this specific call.
        """
        required = list(REQUIRED_FIELDS.get(table_key, []))
        if extra_required:
            required = list(set(required + extra_required))

        total = len(df)
        warnings: List[str] = []
        errors: List[str] = []
        missing_required: Dict[str, int] = {}
        null_pct: Dict[str, float] = {}
        type_errors: Dict[str, int] = {}

        # ── 1. Missing required fields ─────────────────────────────────
        for field in required:
            if field not in df.columns:
                missing_required[field] = total
                errors.append(f"Required field '{field}' is missing entirely.")

        # ── 2. Null analysis ──────────────────────────────────────────
        for col in df.columns:
            if col.startswith("_ext_"):
                continue
            n_null = df[col].isna().sum()
            pct = n_null / total if total > 0 else 0.0
            null_pct[col] = round(float(pct), 4)

            if col in required and pct > 0:
                errors.append(f"Required field '{col}' has {n_null} nulls ({pct:.1%}).")
            elif pct >= self.null_error_threshold:
                errors.append(f"Field '{col}' is {pct:.1%} null (≥{self.null_error_threshold:.0%}).")
            elif pct >= self.null_warn_threshold:
                warnings.append(f"Field '{col}' is {pct:.1%} null.")

        # ── 3. Duplicate detection ─────────────────────────────────────
        id_field = self._id_field(table_key, df)
        n_dupes = 0
        if id_field and id_field in df.columns:
            n_dupes = int(df.duplicated(subset=[id_field]).sum())
            if n_dupes > 0:
                warnings.append(f"{n_dupes} duplicate records detected on '{id_field}'.")

        # ── 4. Numeric type sanity ─────────────────────────────────────
        numeric_hints = {
            "anomaly_score": (0, 1),
            "confidence": (0, 1),
            "utilization_pct": (0, 100),
            "downtime_minutes": (0, None),
            "cycle_time_sec": (0, None),
            "duration_minutes": (0, None),
        }
        for col, (lo, hi) in numeric_hints.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            n_bad = series.isna().sum() - df[col].isna().sum()
            if n_bad > 0:
                type_errors[col] = int(n_bad)
                errors.append(f"Field '{col}' has {n_bad} non-numeric values.")
            if hi is not None:
                out_range = ((series < lo) | (series > hi)).sum()
                if out_range > 0:
                    warnings.append(
                        f"Field '{col}' has {out_range} values outside [{lo}, {hi}]."
                    )

        # ── 5. Timestamp parsing ────────────────────────────────────────
        for col in df.columns:
            if "timestamp" in col or col in ("start_time", "end_time"):
                n_bad = pd.to_datetime(df[col], errors="coerce").isna().sum() - df[col].isna().sum()
                if n_bad > 0:
                    warnings.append(f"Timestamp field '{col}' has {n_bad} unparseable values.")

        # ── Build report ───────────────────────────────────────────────
        n_invalid = len(errors)   # approximate; each error may cover multiple rows
        is_valid = len(errors) == 0

        report = DataQualityReport(
            total_records=total,
            valid_records=max(0, total - n_dupes),
            invalid_records=n_dupes,
            missing_required_fields=missing_required,
            null_pct_by_field=null_pct,
            duplicate_records=n_dupes,
            type_errors=type_errors,
            warnings=warnings,
            errors=errors,
            is_valid=is_valid,
        )

        log_fn = logger.warning if not is_valid else logger.info
        log_fn("Validation of '%s': %d records, %d errors, %d warnings.",
               table_key, total, len(errors), len(warnings))

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _id_field(table_key: str, df: pd.DataFrame) -> Optional[str]:
        candidates = {
            "inspection": "unit_id",
            "products":   "unit_id",
            "economics":  "unit_id",
            "stations":   "station_id",
            "batches":    "batch_id",
            "downtime":   "downtime_id",
        }
        field = candidates.get(table_key)
        return field if (field and field in df.columns) else None
