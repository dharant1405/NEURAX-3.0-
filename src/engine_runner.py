"""
src/engine_runner.py

Orchestrates all InspectIQ engines.
Used by the Streamlit dashboard as an in-process engine coordinator.
Handles loading, training, scoring, and result caching in session state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.economics.economics_engine import EconomicsEngine
from src.flow.flow_engine import FlowEngine
from src.ingestion.data_loader import DataLoader
from src.ingestion.schema_adapter import SchemaAdapter
from src.ingestion.validator import DataValidator
from src.recommend.recommendation_engine import RecommendationEngine
from src.rootcause.rootcause_engine import RootCauseEngine
from src.uncertainty.decision_engine import DecisionEngine
from src.vision.vision_engine import VisionEngine

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent


class EngineRunner:
    """
    Coordinates all InspectIQ engines.
    Results are cached in Streamlit session_state for interactive use.
    """

    def __init__(self, config_dir: Optional[str | Path] = None):
        cfg_dir = Path(config_dir) if config_dir else _ROOT.parent / "configs"

        self.adapter = SchemaAdapter.from_yaml(cfg_dir / "schema_mapping.yaml")
        self.validator = DataValidator()
        self.loader = DataLoader(self.adapter, self.validator)
        self.vision = VisionEngine(cfg_dir / "vision.yaml")
        self.decision = DecisionEngine(cfg_dir / "uncertainty.yaml")
        self.rootcause = RootCauseEngine(cfg_dir / "rootcause.yaml")
        self.flow = FlowEngine(cfg_dir / "flow.yaml")
        self.economics = EconomicsEngine(cfg_dir / "economics.yaml")
        self.recommend = RecommendationEngine(cfg_dir / "simulation.yaml")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_from_directory(self, directory: str | Path) -> Dict[str, Any]:
        """Load all dataset tables from a directory into session state."""
        tables, reports = self.loader.load_directory(directory)
        self._store_tables(tables)
        st.session_state["quality_reports"] = reports
        st.session_state["data_loaded"] = True
        return {"tables": list(tables.keys()), "reports": reports}

    def load_uploaded_files(self, uploaded_files: List) -> Dict[str, Any]:
        """
        Load from Streamlit uploaded file objects.
        Tries to infer the table key from the filename.
        """
        import io
        tables = {}
        reports = {}

        for f in uploaded_files:
            # Infer table key from filename (e.g. "inspection.csv" → "inspection")
            stem = Path(f.name).stem.lower()
            table_key = self._infer_table_key(stem)

            try:
                raw_df = pd.read_csv(io.BytesIO(f.read()))
                if table_key == "process_parameters":
                    adapted = self.adapter.adapt_process_parameters(raw_df)
                else:
                    adapted = self.adapter.adapt(table_key, raw_df)
                report = self.validator.validate(table_key, adapted)
                tables[table_key] = adapted
                reports[table_key] = report
                logger.info("Loaded uploaded file '%s' as table '%s' (%d rows)",
                            f.name, table_key, len(adapted))
            except Exception as exc:
                logger.error("Failed to load '%s': %s", f.name, exc)

        self._store_tables(tables)
        st.session_state.setdefault("quality_reports", {}).update(reports)
        if tables:
            st.session_state["data_loaded"] = True
        return {"tables": list(tables.keys()), "reports": reports}

    # ------------------------------------------------------------------
    # Vision + Decision
    # ------------------------------------------------------------------

    def run_vision(self) -> Optional[pd.DataFrame]:
        """Run vision engine + decision assignment on loaded inspection data."""
        inspection_df = st.session_state.get("inspection_df")
        process_df = st.session_state.get("process_parameters_df")

        if inspection_df is None or len(inspection_df) == 0:
            logger.warning("No inspection data loaded.")
            return None

        # Train on normal records
        self.vision.fit(inspection_df, process_df)

        # Train classifier if labels exist
        if "defect_family" in inspection_df.columns:
            self.vision.fit_classifier(inspection_df, process_df)

        # Score all records
        scored_df = self.vision.score(inspection_df, process_df)

        # Calibrate decision thresholds
        self.decision.calibrate(scored_df)

        # Assign decision states
        result_df = self.decision.assign_decisions(scored_df)

        st.session_state["vision_results"] = result_df
        st.session_state["vision_run"] = True
        st.session_state["decision_summary"] = self.decision.summarize(result_df)
        return result_df

    # ------------------------------------------------------------------
    # Root-cause analysis
    # ------------------------------------------------------------------

    def run_rootcause(self) -> List[Dict]:
        """Run root-cause analysis on inspection + process data."""
        inspection_df = st.session_state.get("inspection_df")
        process_df = st.session_state.get("process_parameters_df")
        production_df = st.session_state.get("production_df")

        if inspection_df is None:
            return []

        hypotheses = self.rootcause.analyze(inspection_df, process_df, production_df)
        st.session_state["hypotheses"] = hypotheses
        st.session_state["rootcause_run"] = True

        # Optional SHAP analysis
        if process_df is not None:
            shap_result = self.rootcause.run_shap_analysis(inspection_df, process_df)
            st.session_state["shap_result"] = shap_result

        return hypotheses

    # ------------------------------------------------------------------
    # Flow analysis
    # ------------------------------------------------------------------

    def run_flow(self) -> Dict[str, Any]:
        """Run bottleneck detection and baseline simulation."""
        production_df = st.session_state.get("production_df")
        stations_df = st.session_state.get("stations_df")
        economics_df = st.session_state.get("economics_df")

        if production_df is None:
            return {}

        # Bottleneck detection
        bottlenecks = self.flow.detect_bottlenecks(production_df, stations_df)
        st.session_state["bottleneck_results"] = bottlenecks

        # Loss calculation
        losses = self.flow.calculate_losses(production_df, economics_df)
        st.session_state["flow_losses"] = losses

        # Baseline simulation
        baseline = self.flow.simulate(stations_df, production_df, label="Baseline")
        st.session_state["baseline_simulation"] = baseline

        st.session_state["flow_run"] = True
        return {"bottlenecks": bottlenecks, "losses": losses, "baseline": baseline}

    # ------------------------------------------------------------------
    # Economics analysis
    # ------------------------------------------------------------------

    def run_economics(self) -> Dict[str, Any]:
        """Run economics analysis on loaded economics data."""
        economics_df = st.session_state.get("economics_df")
        inspection_df = st.session_state.get("inspection_df")

        if economics_df is None:
            logger.warning("No economics data loaded.")
            return {}

        summary = self.economics.calculate(economics_df, inspection_df)
        st.session_state["economics_summary"] = summary
        st.session_state["economics_run"] = True

        # Threshold optimization (if vision results available)
        vision_df = st.session_state.get("vision_results")
        if vision_df is not None and "anomaly_score_final" in vision_df.columns:
            scores = vision_df["anomaly_score_final"].fillna(0.5).values
            labels = (vision_df.get("inspection_result", pd.Series()) == "defective").astype(int).values
            threshold_result = self.economics.optimize_threshold(scores, labels)
            st.session_state["threshold_optimization"] = threshold_result

            # Feed optimized thresholds back to decision engine
            self.decision.update_thresholds_from_economics(
                threshold_result["optimal_low_threshold"],
                threshold_result["optimal_high_threshold"],
            )

        return summary

    # ------------------------------------------------------------------
    # Scenario simulation
    # ------------------------------------------------------------------

    def run_scenario(self, scenario_params: Dict) -> Dict[str, Any]:
        """Run a what-if scenario and compute delta vs baseline."""
        stations_df = st.session_state.get("stations_df")
        production_df = st.session_state.get("production_df")
        economics_summary = st.session_state.get("economics_summary", {})
        baseline = st.session_state.get("baseline_simulation", {})

        scenario_result = self.flow.simulate(
            stations_df, production_df, scenario_params=scenario_params,
            label="Scenario"
        )
        st.session_state["scenario_simulation"] = scenario_result

        # Compute economic delta
        if baseline and economics_summary:
            delta = self.economics.compute_scenario_delta(baseline, scenario_result, economics_summary)
            st.session_state["scenario_delta"] = delta
        else:
            delta = {}

        return {"scenario": scenario_result, "delta": delta}

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def run_recommendations(self) -> List[Dict]:
        """Generate advisory recommendations from all engine outputs."""
        hypotheses = st.session_state.get("hypotheses", [])
        bottlenecks = st.session_state.get("bottleneck_results", [])
        economics_summary = st.session_state.get("economics_summary", {})
        scenario_delta = st.session_state.get("scenario_delta")

        recs = self.recommend.generate(
            hypotheses, bottlenecks, economics_summary, scenario_delta
        )
        st.session_state["recommendations"] = recs
        return recs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _store_tables(self, tables: Dict[str, pd.DataFrame]):
        """Store adapted tables in session state with canonical key names."""
        key_map = {
            "inspection": "inspection_df",
            "products": "products_df",
            "process_parameters": "process_parameters_df",
            "production": "production_df",
            "downtime": "downtime_df",
            "economics": "economics_df",
            "stations": "stations_df",
            "batches": "batches_df",
            "defect_catalog": "defect_catalog_df",
        }
        for table_key, df in tables.items():
            session_key = key_map.get(table_key, f"{table_key}_df")
            st.session_state[session_key] = df
            logger.debug("Stored table '%s' (%d rows) as '%s'", table_key, len(df), session_key)

    @staticmethod
    def _infer_table_key(stem: str) -> str:
        """Guess table key from filename stem."""
        for key in ["inspection", "products", "process_parameters", "process",
                    "production", "downtime", "economics", "stations",
                    "batches", "defect_catalog", "defects"]:
            if key.replace("_", "") in stem.replace("_", "").replace("-", ""):
                return key if key != "process" else "process_parameters"
        return "inspection"  # fallback
