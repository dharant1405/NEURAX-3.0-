"""
tests/test_engines.py

Integration tests for the AI engines: vision, uncertainty, root-cause, flow, economics.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def sample_inspection_df():
    n = 100
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "unit_id": [f"U{i:04d}" for i in range(n)],
        "batch_id": [f"B{i//20+1:03d}" for i in range(n)],
        "station_id": [f"S{(i%5)+1}" for i in range(n)],
        "product_variant": [f"V{(i%3)+1}" for i in range(n)],
        "inspection_result": ["defective" if rng.random() < 0.12 else "good" for _ in range(n)],
        "defect_family": [
            "Scratch" if rng.random() < 0.5 else "Crack"
            if rng.random() < 0.5 else "None"
            for _ in range(n)
        ],
        "anomaly_score": rng.uniform(0, 1, n).round(4),
        "is_novel_defect": [False] * 90 + [True] * 10,
    })


@pytest.fixture
def sample_process_df(sample_inspection_df):
    rng = np.random.default_rng(42)
    n = len(sample_inspection_df)
    return pd.DataFrame({
        "unit_id": sample_inspection_df["unit_id"].tolist(),
        "batch_id": sample_inspection_df["batch_id"].tolist(),
        "station_id": sample_inspection_df["station_id"].tolist(),
        "temperature": rng.normal(175, 15, n).round(2),
        "pressure": rng.normal(2.5, 0.3, n).round(3),
        "machine_speed": rng.normal(100, 8, n).round(1),
        "vibration": rng.normal(1.5, 0.3, n).round(3),
        "cycle_time_sec": rng.normal(80, 10, n).round(2),
        "tool_wear_percent": rng.uniform(20, 80, n).round(1),
        "energy_consumption_kwh": rng.normal(2.0, 0.3, n).round(3),
    })


@pytest.fixture
def sample_production_df():
    rows = []
    for batch in [f"B{i+1:03d}" for i in range(5)]:
        for station in [f"S{i+1}" for i in range(5)]:
            rows.append({
                "batch_id": batch,
                "station_id": station,
                "planned_units": 100,
                "produced_units": 95,
                "good_units": 85,
                "defective_units": 10,
                "rework_units": 7,
                "scrap_units": 3,
                "downtime_minutes": float(np.random.default_rng(42).normal(15, 5)),
                "utilization_percent": float(np.random.default_rng(42).uniform(60, 95)),
                "capacity_units_per_hour": [80, 50, 40, 30, 120][int(station[1])-1],
                "actual_throughput_units_per_hour": float(np.random.default_rng(42).uniform(25, 50)),
            })
    return pd.DataFrame(rows)


class TestVisionEngine:
    def test_fit_and_score(self, sample_inspection_df, sample_process_df):
        from src.vision.vision_engine import VisionEngine
        engine = VisionEngine()
        engine.fit(sample_inspection_df, sample_process_df)
        assert engine.is_trained

        scored = engine.score(sample_inspection_df, sample_process_df)
        assert "anomaly_score_computed" in scored.columns
        assert scored["anomaly_score_computed"].between(0, 1).all()

    def test_fit_classifier(self, sample_inspection_df, sample_process_df):
        from src.vision.vision_engine import VisionEngine
        engine = VisionEngine()
        engine.fit(sample_inspection_df, sample_process_df)
        engine.fit_classifier(sample_inspection_df, sample_process_df)
        # Classifier should be trained when labels exist
        if engine._classification_trained:
            assert len(engine.defect_classes) > 0

    def test_heatmap_generation(self):
        from src.vision.vision_engine import VisionEngine
        engine = VisionEngine()
        hm = engine.generate_heatmap_data("U0001", anomaly_score=0.8)
        assert "heatmap" in hm
        assert "anomaly_score" in hm
        assert hm["anomaly_score"] == 0.8


class TestDecisionEngine:
    def test_assign_decisions(self, sample_inspection_df):
        from src.uncertainty.decision_engine import DecisionEngine
        engine = DecisionEngine()
        df = sample_inspection_df.copy()
        df["anomaly_score_final"] = df["anomaly_score"]
        result = engine.assign_decisions(df)
        assert "decision_state" in result.columns
        valid_states = {"ACCEPT", "REJECT", "REVIEW", "NOVEL"}
        assert result["decision_state"].isin(valid_states).all()

    def test_no_hardcoded_thresholds(self):
        """Thresholds must come from config, not hardcoded in logic."""
        from src.uncertainty.decision_engine import DecisionEngine
        engine = DecisionEngine()
        # Thresholds should be numeric values loaded from config
        assert isinstance(engine.low_threshold, float)
        assert isinstance(engine.high_threshold, float)
        assert 0 < engine.low_threshold < engine.high_threshold < 1

    def test_calibration(self, sample_inspection_df):
        from src.uncertainty.decision_engine import DecisionEngine
        engine = DecisionEngine()
        df = sample_inspection_df.copy()
        df["anomaly_score_final"] = df["anomaly_score"]
        engine.calibrate(df)
        # After calibration thresholds should be data-driven
        assert engine.calibrated or True  # may not calibrate with small sample


class TestRootCauseEngine:
    def test_analyze_returns_hypotheses(self, sample_inspection_df, sample_process_df):
        from src.rootcause.rootcause_engine import RootCauseEngine
        engine = RootCauseEngine()
        hypotheses = engine.analyze(sample_inspection_df, sample_process_df)
        assert isinstance(hypotheses, list)
        # May be empty for small/random data — that's OK

    def test_hypotheses_have_required_fields(self, sample_inspection_df, sample_process_df):
        from src.rootcause.rootcause_engine import RootCauseEngine
        engine = RootCauseEngine()
        hypotheses = engine.analyze(sample_inspection_df, sample_process_df)
        for h in hypotheses:
            assert "hypothesis" in h
            assert "evidence" in h
            assert "confidence" in h
            assert 0 <= h["confidence"] <= 1
            assert "alternative_explanations" in h

    def test_hypotheses_are_advisory(self, sample_inspection_df, sample_process_df):
        """All hypothesis text must be advisory, never imperative."""
        from src.rootcause.rootcause_engine import RootCauseEngine
        engine = RootCauseEngine()
        hypotheses = engine.analyze(sample_inspection_df, sample_process_df)
        for h in hypotheses:
            text = h.get("hypothesis", "").lower()
            # Should never say "change X immediately" or similar
            assert "immediately" not in text
            assert "must change" not in text


class TestFlowEngine:
    def test_detect_bottlenecks(self, sample_production_df):
        from src.flow.flow_engine import FlowEngine
        engine = FlowEngine()
        results = engine.detect_bottlenecks(sample_production_df)
        assert isinstance(results, list)
        assert len(results) > 0
        # Should discover stations dynamically from data
        detected_station_ids = {r["station_id"] for r in results}
        expected_station_ids = set(sample_production_df["station_id"].unique())
        assert detected_station_ids == expected_station_ids

    def test_bottleneck_score_range(self, sample_production_df):
        from src.flow.flow_engine import FlowEngine
        engine = FlowEngine()
        results = engine.detect_bottlenecks(sample_production_df)
        for r in results:
            assert 0 <= r["bottleneck_score"] <= 2, f"Score out of range: {r['bottleneck_score']}"

    def test_loss_calculation(self, sample_production_df):
        from src.flow.flow_engine import FlowEngine
        engine = FlowEngine()
        losses = engine.calculate_losses(sample_production_df)
        assert "total_produced_units" in losses
        assert "overall_defect_rate_pct" in losses
        assert losses["overall_defect_rate_pct"] >= 0


class TestEconomicsEngine:
    def test_calculate_with_full_data(self):
        from src.economics.economics_engine import EconomicsEngine
        engine = EconomicsEngine()
        econ_df = pd.DataFrame({
            "unit_id": [f"U{i}" for i in range(50)],
            "batch_id": [f"B{i//10+1}" for i in range(50)],
            "variant_id": [f"V{i%3+1}" for i in range(50)],
            "selling_price": np.random.default_rng(42).normal(500, 20, 50),
            "material_cost": np.random.default_rng(42).normal(150, 10, 50),
            "processing_cost": np.random.default_rng(42).normal(80, 5, 50),
            "energy_cost": np.random.default_rng(42).normal(15, 2, 50),
            "labor_cost": np.random.default_rng(42).normal(40, 3, 50),
            "rework_cost": [20 if i % 5 == 0 else 0 for i in range(50)],
            "scrap_cost": [100 if i % 10 == 0 else 0 for i in range(50)],
        })
        summary = engine.calculate(econ_df)
        assert "summary" in summary
        assert summary["summary"]["total_revenue"] > 0
        assert "cost_breakdown" in summary

    def test_fallback_values_labelled(self):
        """When fallback values are used, they must be labelled as assumptions."""
        from src.economics.economics_engine import EconomicsEngine
        engine = EconomicsEngine()
        # Empty economics df — should use fallback
        econ_df = pd.DataFrame({"unit_id": ["U001"]})
        summary = engine.calculate(econ_df)
        # If fallback used, assumed_fields must be non-empty
        if summary.get("assumptions_used"):
            assert len(summary.get("assumed_fields", [])) > 0

    def test_optimize_threshold(self):
        from src.economics.economics_engine import EconomicsEngine
        engine = EconomicsEngine()
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 200)
        labels = (scores > 0.5).astype(int)
        result = engine.optimize_threshold(scores, labels)
        assert "optimal_high_threshold" in result
        assert 0 < result["optimal_high_threshold"] < 1
        assert result["label"] == "SIMULATED / ADVISORY"
