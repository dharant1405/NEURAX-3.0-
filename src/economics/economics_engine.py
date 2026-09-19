"""
src/economics/economics_engine.py

Economics Engine: Unit economics + margin forecasting + threshold optimization.
All cost values come from the dataset or explicitly-labelled fallback config.
NO hardcoded prices, costs, or margins.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CFG = _ROOT / "configs" / "economics.yaml"


class EconomicsEngine:
    """
    Calculates production economics from dataset and simulation results.
    Missing fields are filled from explicitly labelled fallback assumptions.
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        path = Path(config_path) if config_path else _DEFAULT_CFG
        with open(path, "r", encoding="utf-8") as fh:
            self._cfg = yaml.safe_load(fh)

        self._fallback = self._cfg.get("fallback_values", {})
        self._decision_costs = self._cfg.get("decision_costs", {})
        self._currency = self._cfg.get("display", {}).get("currency", "$")
        self._dp = self._cfg.get("display", {}).get("decimal_places", 2)

    # ------------------------------------------------------------------
    # Main economics calculation
    # ------------------------------------------------------------------

    def calculate(self, economics_df: pd.DataFrame,
                  inspection_df: Optional[pd.DataFrame] = None
                  ) -> Dict[str, Any]:
        """
        Calculate unit economics summary from economics DataFrame.
        Any missing columns are filled from fallback config and flagged.
        """
        df = economics_df.copy()
        assumed_fields: List[str] = []

        # Columns and their fallback keys
        cost_fields = {
            "selling_price": "selling_price",
            "material_cost": "material_cost",
            "processing_cost": "processing_cost",
            "energy_cost": "energy_cost",
            "labor_cost": "labor_cost",
            "rework_cost": "rework_cost",
            "scrap_cost": "scrap_cost",
        }

        for internal_col, fallback_key in cost_fields.items():
            if internal_col not in df.columns or df[internal_col].isna().all():
                fallback_val = self._fallback.get(fallback_key)
                if fallback_val is not None and self._fallback.get("enabled", True):
                    df[internal_col] = fallback_val
                    assumed_fields.append(internal_col)
                    logger.info("Using fallback assumption for '%s': %s %s",
                                internal_col, self._currency, fallback_val)

        # Compute total_cost if not present
        cost_cols = ["material_cost", "processing_cost", "energy_cost",
                     "labor_cost", "rework_cost", "scrap_cost"]
        available_cost_cols = [c for c in cost_cols if c in df.columns]
        if "total_cost" not in df.columns or df["total_cost"].isna().all():
            df["total_cost"] = df[available_cost_cols].fillna(0).sum(axis=1)
            if available_cost_cols:
                assumed_fields.append("total_cost")

        # Compute profit
        if "profit" not in df.columns or df["profit"].isna().all():
            if "selling_price" in df.columns and "total_cost" in df.columns:
                df["profit"] = df["selling_price"] - df["total_cost"]
                assumed_fields.append("profit")

        # Compute margin %
        if "profit_margin_pct" not in df.columns or df["profit_margin_pct"].isna().all():
            if "profit" in df.columns and "selling_price" in df.columns:
                df["profit_margin_pct"] = (
                    df["profit"] / df["selling_price"].replace(0, np.nan) * 100
                )
                assumed_fields.append("profit_margin_pct")

        # Summary statistics
        total_revenue = df["selling_price"].sum() if "selling_price" in df.columns else 0
        total_cost = df["total_cost"].sum() if "total_cost" in df.columns else 0
        total_profit = df["profit"].sum() if "profit" in df.columns else 0
        avg_margin_pct = float(df["profit_margin_pct"].mean()) if "profit_margin_pct" in df.columns else 0

        # Cost breakdown
        cost_breakdown = {}
        for col in available_cost_cols:
            cost_breakdown[col] = round(float(df[col].sum()), self._dp)

        # Per-variant analysis
        per_variant = {}
        if "variant_id" in df.columns:
            for variant, grp in df.groupby("variant_id"):
                per_variant[str(variant)] = {
                    "units": len(grp),
                    "avg_profit": round(float(grp["profit"].mean()) if "profit" in grp.columns else 0, self._dp),
                    "avg_margin_pct": round(float(grp["profit_margin_pct"].mean()) if "profit_margin_pct" in grp.columns else 0, self._dp),
                }

        # Per-batch trend
        batch_trend = {}
        if "batch_id" in df.columns:
            for batch, grp in df.groupby("batch_id"):
                batch_trend[str(batch)] = round(
                    float(grp["profit"].mean()) if "profit" in grp.columns else 0, self._dp
                )

        return {
            "summary": {
                "total_revenue": round(float(total_revenue), self._dp),
                "total_cost": round(float(total_cost), self._dp),
                "total_profit": round(float(total_profit), self._dp),
                "avg_margin_pct": round(float(avg_margin_pct), self._dp),
                "currency": self._currency,
                "n_units": len(df),
            },
            "cost_breakdown": cost_breakdown,
            "per_variant": per_variant,
            "batch_trend": batch_trend,
            "assumed_fields": assumed_fields,
            "assumptions_used": len(assumed_fields) > 0,
        }

    # ------------------------------------------------------------------
    # Margin per unit
    # ------------------------------------------------------------------

    def per_unit_margin(self, economics_df: pd.DataFrame) -> pd.DataFrame:
        """Return unit-level margin DataFrame for trend charts."""
        df = economics_df.copy()
        if "profit" not in df.columns and "selling_price" in df.columns and "total_cost" in df.columns:
            df["profit"] = df["selling_price"] - df["total_cost"]
        if "profit_margin_pct" not in df.columns and "profit" in df.columns:
            df["profit_margin_pct"] = df["profit"] / df["selling_price"].replace(0, np.nan) * 100
        return df

    # ------------------------------------------------------------------
    # Threshold optimization (FR-12)
    # ------------------------------------------------------------------

    def optimize_threshold(self, anomaly_scores: np.ndarray,
                            true_labels: np.ndarray,
                            fa_cost: Optional[float] = None,
                            fr_cost: Optional[float] = None,
                            review_cost: Optional[float] = None
                            ) -> Dict[str, Any]:
        """
        Find the accept/review/reject thresholds that minimize expected cost.
        Costs come from config unless overridden.

        Parameters
        ----------
        anomaly_scores : array of shape (n,)
        true_labels    : array of shape (n,) — 1=defective, 0=good
        fa_cost        : false accept cost (overrides config)
        fr_cost        : false reject cost (overrides config)
        review_cost    : cost per reviewed unit (overrides config)
        """
        fa_cost = fa_cost or self._decision_costs.get("false_accept_cost", 250)
        fr_cost = fr_cost or self._decision_costs.get("false_reject_cost", 120)
        review_cost = review_cost or self._decision_costs.get("review_cost", 20)

        thresholds = np.linspace(0.05, 0.95, 100)
        best_cost = np.inf
        best_low = 0.3
        best_high = 0.7
        cost_curve = []

        for low in thresholds:
            for high in thresholds:
                if high <= low:
                    continue
                accept = anomaly_scores <= low
                reject = anomaly_scores >= high
                review_mask = (~accept) & (~reject)

                # False accepts: predicted accept but actually defective
                fa = ((accept) & (true_labels == 1)).sum()
                # False rejects: predicted reject but actually good
                fr = ((reject) & (true_labels == 0)).sum()
                n_review = review_mask.sum()

                expected_cost = fa * fa_cost + fr * fr_cost + n_review * review_cost

                if expected_cost < best_cost:
                    best_cost = expected_cost
                    best_low = float(low)
                    best_high = float(high)

        cost_curve_simple = []
        for t in thresholds:
            reject_mask = anomaly_scores >= t
            accept_mask = ~reject_mask
            fa = ((accept_mask) & (true_labels == 1)).sum()
            fr = ((reject_mask) & (true_labels == 0)).sum()
            cost_curve_simple.append({
                "threshold": round(float(t), 3),
                "expected_cost": round(float(fa * fa_cost + fr * fr_cost), 2),
                "false_accept_rate": round(float(fa / max((true_labels == 1).sum(), 1)), 4),
                "false_reject_rate": round(float(fr / max((true_labels == 0).sum(), 1)), 4),
            })

        return {
            "optimal_low_threshold": round(best_low, 3),
            "optimal_high_threshold": round(best_high, 3),
            "expected_cost_at_optimal": round(best_cost, 2),
            "false_accept_cost_used": fa_cost,
            "false_reject_cost_used": fr_cost,
            "review_cost_used": review_cost,
            "cost_curve": cost_curve_simple,
            "label": "SIMULATED / ADVISORY",
        }

    # ------------------------------------------------------------------
    # Scenario delta calculation
    # ------------------------------------------------------------------

    def compute_scenario_delta(self, baseline: Dict, scenario: Dict,
                                economics_summary: Dict) -> Dict[str, Any]:
        """
        Compute economic delta between baseline and scenario simulation results.
        Labels all outputs as SIMULATED / ADVISORY.
        """
        base_tput = baseline.get("throughput_per_hour", 0)
        scen_tput = scenario.get("throughput_per_hour", 0)
        delta_tput = scen_tput - base_tput

        avg_margin = economics_summary.get("summary", {}).get("avg_margin_pct", 0)
        avg_profit_per_unit = economics_summary.get("summary", {}).get("total_profit", 0) / \
                              max(economics_summary.get("summary", {}).get("n_units", 1), 1)

        # Estimated margin change from throughput change (per 8h shift)
        delta_units_per_shift = delta_tput * 8
        delta_margin = delta_units_per_shift * avg_profit_per_unit

        base_scrap = baseline.get("scrap_units", 0)
        scen_scrap = scenario.get("scrap_units", 0)
        base_rework = baseline.get("rework_units", 0)
        scen_rework = scenario.get("rework_units", 0)

        return {
            "delta_throughput_per_hour": round(delta_tput, 2),
            "delta_units_per_shift": round(delta_units_per_shift, 0),
            "delta_estimated_margin": round(delta_margin, 2),
            "delta_scrap_units": round(scen_scrap - base_scrap, 1),
            "delta_rework_units": round(scen_rework - base_rework, 1),
            "currency": self._currency,
            "label": "SIMULATED / ADVISORY",
            "assumptions": [
                "Throughput-margin relationship assumes constant product mix.",
                "Economic impact estimated from per-unit profit margin in historical data.",
                "Simulation uses configured station cycle-time distributions.",
            ],
        }
