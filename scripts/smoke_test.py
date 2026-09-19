"""
scripts/smoke_test.py

Quick smoke test to verify all engines load and produce output.
Run: python scripts/smoke_test.py
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import logging
import tempfile
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_test")


def test_synthetic_generation():
    logger.info("=== Test 1: Synthetic Data Generation ===")
    from data.synthetic.generator import SyntheticDataGenerator, DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG.copy()
    cfg["n_units"] = 200
    cfg["n_batches"] = 3
    gen = SyntheticDataGenerator(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        tables = gen.generate_all(tmpdir)
        for name, df in tables.items():
            logger.info(f"  ✓ {name}: {len(df)} rows")
        assert "inspection" in tables
        assert "production" in tables
        logger.info("  PASS: All tables generated")
        return tables, tmpdir


def test_schema_adapter():
    logger.info("=== Test 2: Schema Adapter ===")
    from src.ingestion.schema_adapter import SchemaAdapter
    adapter = SchemaAdapter.from_yaml()
    raw = pd.DataFrame({
        "product_id": ["U001"],
        "inspection_result": ["good"],
        "anomaly_score": [0.1],
    })
    adapted = adapter.adapt("inspection", raw)
    assert "unit_id" in adapted.columns
    logger.info("  ✓ Adapter maps product_id → unit_id")
    logger.info("  PASS")


def test_vision_engine():
    logger.info("=== Test 3: Vision Engine ===")
    from src.vision.vision_engine import VisionEngine
    engine = VisionEngine()
    rng = np.random.default_rng(42)
    n = 100
    insp_df = pd.DataFrame({
        "unit_id": [f"U{i}" for i in range(n)],
        "inspection_result": ["defective" if rng.random() < 0.1 else "good" for _ in range(n)],
        "defect_family": ["Scratch" if rng.random() < 0.1 else "None" for _ in range(n)],
        "batch_id": [f"B{i//50+1}" for i in range(n)],
    })
    proc_df = pd.DataFrame({
        "unit_id": [f"U{i}" for i in range(n)],
        "temperature": rng.normal(175, 15, n),
        "vibration": rng.normal(1.5, 0.3, n),
        "cycle_time_sec": rng.normal(80, 10, n),
    })
    engine.fit(insp_df, proc_df)
    scored = engine.score(insp_df, proc_df)
    assert "anomaly_score_computed" in scored.columns
    logger.info(f"  ✓ Scored {len(scored)} units")
    logger.info(f"  ✓ Score range: [{scored['anomaly_score_computed'].min():.3f}, {scored['anomaly_score_computed'].max():.3f}]")
    logger.info("  PASS")
    return insp_df, proc_df, scored


def test_decision_engine(scored_df):
    logger.info("=== Test 4: Decision Engine ===")
    from src.uncertainty.decision_engine import DecisionEngine
    engine = DecisionEngine()
    df = scored_df.copy()
    df["anomaly_score_final"] = df["anomaly_score_computed"]
    result = engine.assign_decisions(df)
    counts = result["decision_state"].value_counts().to_dict()
    logger.info(f"  ✓ Decision counts: {counts}")
    assert all(s in {"ACCEPT", "REJECT", "REVIEW", "NOVEL"} for s in counts)
    logger.info("  PASS")


def test_rootcause_engine(insp_df, proc_df):
    logger.info("=== Test 5: Root-Cause Engine ===")
    from src.rootcause.rootcause_engine import RootCauseEngine
    engine = RootCauseEngine()
    hypotheses = engine.analyze(insp_df, proc_df)
    logger.info(f"  ✓ Generated {len(hypotheses)} hypotheses")
    for h in hypotheses[:2]:
        logger.info(f"    [{h.get('rank')}] {h.get('defect_family')}: confidence={h.get('confidence'):.2f}")
    logger.info("  PASS")


def test_flow_engine():
    logger.info("=== Test 6: Flow Engine ===")
    from src.flow.flow_engine import FlowEngine
    engine = FlowEngine()
    rng = np.random.default_rng(42)
    prod_df = pd.DataFrame({
        "batch_id": [f"B{i}" for i in range(1, 4)] * 4,
        "station_id": ["S1", "S2", "S3", "S4"] * 3,
        "produced_units": rng.integers(80, 100, 12).tolist(),
        "good_units": rng.integers(70, 90, 12).tolist(),
        "defective_units": rng.integers(5, 15, 12).tolist(),
        "rework_units": rng.integers(2, 8, 12).tolist(),
        "scrap_units": rng.integers(1, 5, 12).tolist(),
        "downtime_minutes": rng.normal(15, 5, 12).tolist(),
        "utilization_percent": rng.uniform(60, 95, 12).tolist(),
        "capacity_units_per_hour": [80, 50, 40, 30] * 3,
        "actual_throughput_units_per_hour": rng.uniform(25, 45, 12).tolist(),
        "planned_units": [100] * 12,
    })
    bottlenecks = engine.detect_bottlenecks(prod_df)
    logger.info(f"  ✓ Detected {len(bottlenecks)} stations")
    logger.info(f"  ✓ Primary bottleneck: {bottlenecks[0]['station_id']} (score: {bottlenecks[0]['bottleneck_score']:.3f})")
    losses = engine.calculate_losses(prod_df)
    logger.info(f"  ✓ Defect rate: {losses['overall_defect_rate_pct']:.1f}%")
    logger.info("  PASS")


def test_economics_engine():
    logger.info("=== Test 7: Economics Engine ===")
    from src.economics.economics_engine import EconomicsEngine
    engine = EconomicsEngine()
    rng = np.random.default_rng(42)
    n = 50
    econ_df = pd.DataFrame({
        "unit_id": [f"U{i}" for i in range(n)],
        "batch_id": [f"B{i//25+1}" for i in range(n)],
        "variant_id": [f"V{i%3+1}" for i in range(n)],
        "selling_price": rng.normal(500, 20, n),
        "material_cost": rng.normal(150, 10, n),
        "processing_cost": rng.normal(80, 5, n),
        "labor_cost": rng.normal(40, 3, n),
        "rework_cost": [20 if i%5==0 else 0 for i in range(n)],
        "scrap_cost": [100 if i%10==0 else 0 for i in range(n)],
    })
    summary = engine.calculate(econ_df)
    s = summary["summary"]
    logger.info(f"  ✓ Revenue: {s['currency']}{s['total_revenue']:,.0f}")
    logger.info(f"  ✓ Profit: {s['currency']}{s['total_profit']:,.0f}")
    logger.info(f"  ✓ Avg Margin: {s['avg_margin_pct']:.1f}%")
    logger.info("  PASS")


def test_recommendation_engine():
    logger.info("=== Test 8: Recommendation Engine ===")
    from src.recommend.recommendation_engine import RecommendationEngine
    engine = RecommendationEngine()
    hypotheses = [
        {"rank": 1, "defect_family": "Scratch", "hypothesis": "Statistical analysis indicates...",
         "evidence": ["Test evidence"], "confidence": 0.75, "effect_size": 0.4,
         "alternative_explanations": ["Confounding"], "statistical_support": {}}
    ]
    bottlenecks = [
        {"station_id": "S4", "station_name": "Assembly", "bottleneck_score": 0.8,
         "utilization_pct": 92, "rank": 1, "is_primary_bottleneck": True,
         "throughput_gap_pct": 25, "downtime_minutes_total": 120}
    ]
    economics = {"summary": {"total_profit": 50000, "n_units": 1000,
                             "avg_margin_pct": 15.0, "currency": "₹"}}
    recs = engine.generate(hypotheses, bottlenecks, economics)
    logger.info(f"  ✓ Generated {len(recs)} recommendations")
    for rec in recs:
        assert rec["status"] == "SIMULATED / ADVISORY"
        logger.info(f"    [{rec['rank']}] Confidence: {rec['confidence']:.0%} — {rec['action'][:60]}...")
    logger.info("  PASS")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  InspectIQ Smoke Test")
    print("="*60 + "\n")

    try:
        test_schema_adapter()
        insp_df, proc_df, scored_df = test_vision_engine()
        test_decision_engine(scored_df)
        test_rootcause_engine(insp_df, proc_df)
        test_flow_engine()
        test_economics_engine()
        test_recommendation_engine()
        print("\n" + "="*60)
        print("  ✅ ALL SMOKE TESTS PASSED")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
