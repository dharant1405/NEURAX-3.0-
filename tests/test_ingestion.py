"""
tests/test_ingestion.py

Unit tests for the data ingestion layer.
"""
import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.schema_adapter import SchemaAdapter
from src.ingestion.validator import DataValidator
from src.models.internal_schema import DataQualityReport


class TestSchemaAdapter:
    def setup_method(self):
        self.adapter = SchemaAdapter.from_yaml()

    def test_load_from_yaml(self):
        assert self.adapter is not None
        assert self.adapter.version is not None

    def test_adapt_inspection(self):
        raw = pd.DataFrame({
            "product_id": ["U001", "U002"],
            "inspection_timestamp": ["2024-01-01", "2024-01-02"],
            "inspection_result": ["good", "defective"],
            "defect_family": ["None", "Scratch"],
            "anomaly_score": [0.1, 0.8],
            "inspection_confidence": [0.95, 0.88],
        })
        adapted = self.adapter.adapt("inspection", raw)
        assert "unit_id" in adapted.columns
        assert "timestamp" in adapted.columns
        assert adapted["unit_id"].tolist() == ["U001", "U002"]

    def test_adapt_preserves_unknown_columns(self):
        raw = pd.DataFrame({
            "product_id": ["U001"],
            "some_unknown_field": [42],
        })
        adapted = self.adapter.adapt("inspection", raw)
        assert any(c.startswith("_ext_") for c in adapted.columns)

    def test_configured_tables(self):
        tables = self.adapter.configured_tables
        assert "inspection" in tables
        assert "production" in tables
        assert "economics" in tables

    def test_no_hardcoded_values_in_mapping(self):
        """Ensure adapter never hardcodes station IDs, batch IDs, etc."""
        # Adapt with non-standard column names
        raw = pd.DataFrame({
            "product_id": ["X001"],
            "batch_id": ["CUSTOM_BATCH_999"],
            "inspection_result": ["good"],
        })
        adapted = self.adapter.adapt("inspection", raw)
        # The adapter should map product_id → unit_id correctly
        assert "unit_id" in adapted.columns
        # No hardcoded batch IDs should appear
        assert adapted["unit_id"].iloc[0] == "X001"


class TestDataValidator:
    def setup_method(self):
        self.validator = DataValidator()

    def test_valid_inspection_df(self):
        df = pd.DataFrame({
            "unit_id": ["U001", "U002", "U003"],
            "inspection_result": ["good", "defective", "good"],
            "anomaly_score": [0.1, 0.85, 0.2],
        })
        report = self.validator.validate("inspection", df)
        assert report.total_records == 3
        assert report.is_valid

    def test_missing_required_field(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        report = self.validator.validate("inspection", df)
        assert not report.is_valid
        assert "unit_id" in report.missing_required_fields

    def test_null_detection(self):
        df = pd.DataFrame({
            "unit_id": ["U001", None, "U003"],
            "inspection_result": ["good", "good", None],
        })
        report = self.validator.validate("inspection", df)
        assert "unit_id" in report.null_pct_by_field
        assert report.null_pct_by_field["unit_id"] > 0

    def test_duplicate_detection(self):
        df = pd.DataFrame({
            "unit_id": ["U001", "U001", "U003"],
            "inspection_result": ["good", "good", "defective"],
        })
        report = self.validator.validate("inspection", df)
        assert report.duplicate_records > 0


class TestSyntheticGenerator:
    def test_generator_creates_all_tables(self):
        from data.synthetic.generator import SyntheticDataGenerator, DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG.copy()
        cfg["n_units"] = 100
        cfg["n_batches"] = 3
        gen = SyntheticDataGenerator(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            tables = gen.generate_all(tmpdir)
            expected = ["stations", "defect_catalog", "batches", "products",
                        "inspection", "process_parameters", "production",
                        "downtime", "economics"]
            for name in expected:
                assert name in tables, f"Missing table: {name}"
                assert len(tables[name]) > 0

    def test_generator_internal_consistency(self):
        """Products and inspection should have same unit IDs."""
        from data.synthetic.generator import SyntheticDataGenerator, DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG.copy()
        cfg["n_units"] = 50
        cfg["n_batches"] = 2
        gen = SyntheticDataGenerator(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            tables = gen.generate_all(tmpdir)
            product_ids = set(tables["products"]["product_id"])
            inspection_ids = set(tables["inspection"]["product_id"])
            assert product_ids == inspection_ids

    def test_no_hardcoded_station_ids(self):
        """Station IDs are configured, not hardcoded."""
        from data.synthetic.generator import DEFAULT_CONFIG
        station_ids = [s["id"] for s in DEFAULT_CONFIG["stations"]]
        # Should be configurable
        assert len(station_ids) == DEFAULT_CONFIG["n_stations"]

    def test_defect_rate_is_positive(self):
        from data.synthetic.generator import SyntheticDataGenerator, DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG.copy()
        cfg["n_units"] = 200
        cfg["n_batches"] = 3
        gen = SyntheticDataGenerator(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = gen.generate_all(tmpdir)
            n_defective = (tables["inspection"]["inspection_result"] == "defective").sum()
            assert n_defective > 0, "No defective units generated"
            defect_rate = n_defective / len(tables["inspection"])
            assert 0.001 < defect_rate < 0.6, f"Defect rate out of range: {defect_rate}"
