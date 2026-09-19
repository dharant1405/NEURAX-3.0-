"""
src/rootcause/rootcause_engine.py

Root-Cause Engine: drift detection + association tests + SHAP analysis.
Produces ranked, evidence-backed hypotheses with statistical support.
No hardcoded station IDs, batch IDs, defect names, or parameter names.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CFG = _ROOT / "configs" / "rootcause.yaml"


class RootCauseEngine:
    """
    Links defect patterns with production/process data.
    Outputs ranked hypotheses with evidence and confidence.
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        path = Path(config_path) if config_path else _DEFAULT_CFG
        with open(path, "r", encoding="utf-8") as fh:
            self._cfg = yaml.safe_load(fh)

        self._assoc_cfg = self._cfg["association_tests"]
        self._hyp_cfg = self._cfg["hypothesis"]
        self._drift_cfg = self._cfg["drift_detection"]

    # ------------------------------------------------------------------
    # Main analysis pipeline
    # ------------------------------------------------------------------

    def analyze(self,
                inspection_df: pd.DataFrame,
                process_df: Optional[pd.DataFrame] = None,
                production_df: Optional[pd.DataFrame] = None
                ) -> List[Dict[str, Any]]:
        """
        Run the full root-cause analysis pipeline.

        Returns
        -------
        List of hypothesis dicts, sorted by confidence (descending).
        """
        hypotheses = []

        defect_families = self._get_defect_families(inspection_df)
        if not defect_families:
            logger.info("No defect families found for root-cause analysis.")
            return []

        for family in defect_families:
            family_hyps = self._analyze_family(
                family, inspection_df, process_df, production_df
            )
            hypotheses.extend(family_hyps)

        # Sort by confidence and rank
        hypotheses.sort(key=lambda h: h["confidence"], reverse=True)
        max_h = self._hyp_cfg.get("max_hypotheses", 5) * len(defect_families)
        hypotheses = hypotheses[:max_h]

        for rank, h in enumerate(hypotheses, 1):
            h["rank"] = rank

        return hypotheses

    def _analyze_family(self, family: str, inspection_df: pd.DataFrame,
                        process_df: Optional[pd.DataFrame],
                        production_df: Optional[pd.DataFrame]) -> List[Dict]:
        hyps = []
        defective_mask = (
            (inspection_df.get("defect_family", pd.Series()) == family) &
            (inspection_df.get("inspection_result", pd.Series()) == "defective")
        )
        defective_df = inspection_df[defective_mask]
        normal_df = inspection_df[~defective_mask]

        if len(defective_df) < self._assoc_cfg.get("min_cell_count", 5):
            return []

        # ── 1. Batch association ───────────────────────────────────────
        if "batch_id" in inspection_df.columns:
            hyp = self._test_categorical_association(
                inspection_df, "batch_id", family, "Batch"
            )
            if hyp:
                hyps.append(hyp)

        # ── 2. Station association ─────────────────────────────────────
        if "station_id" in inspection_df.columns:
            hyp = self._test_categorical_association(
                inspection_df, "station_id", family, "Station"
            )
            if hyp:
                hyps.append(hyp)

        # ── 3. Process parameter drift ─────────────────────────────────
        if process_df is not None and len(process_df) > 0:
            param_hyps = self._test_process_parameters(
                inspection_df, process_df, family
            )
            hyps.extend(param_hyps)

        # ── 4. Temporal trend ──────────────────────────────────────────
        if "timestamp" in inspection_df.columns:
            hyp = self._test_temporal_trend(inspection_df, family)
            if hyp:
                hyps.append(hyp)

        return hyps

    # ------------------------------------------------------------------
    # Categorical association (chi-square test)
    # ------------------------------------------------------------------

    def _test_categorical_association(self, df: pd.DataFrame, factor_col: str,
                                       defect_family: str,
                                       factor_label: str) -> Optional[Dict]:
        alpha = self._assoc_cfg.get("alpha", 0.05)
        min_count = self._assoc_cfg.get("min_cell_count", 5)

        # Build contingency table
        is_defect = (df.get("defect_family", pd.Series()) == defect_family) & \
                    (df.get("inspection_result", pd.Series()) == "defective")
        ct = pd.crosstab(df[factor_col], is_defect)

        if ct.shape[0] < 2 or ct.values.min() < min_count:
            return None

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(ct.values)
        except Exception:
            return None

        if p_value > alpha:
            return None

        # Effect size (Cramer's V)
        n = ct.values.sum()
        cramer_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

        # Which factor level has highest defect rate?
        rates = ct.div(ct.sum(axis=1), axis=0)
        if True in rates.columns:
            worst_level = rates[True].idxmax()
            worst_rate = float(rates[True].max())
        else:
            return None

        confidence = float(np.clip(cramer_v * (1 - p_value), 0.1, 0.95))

        return {
            "rank": 0,
            "defect_family": defect_family,
            "hypothesis": (
                f"Simulation of statistical associations indicates that "
                f"{factor_label} '{worst_level}' is associated with elevated "
                f"{defect_family} defect rate ({worst_rate:.1%}) under current dataset conditions."
            ),
            "evidence": [
                f"Chi-square test: χ²={chi2:.2f}, p={p_value:.4f} (n={n})",
                f"Cramér's V effect size: {cramer_v:.3f}",
                f"Highest defect rate at {factor_label} '{worst_level}': {worst_rate:.1%}",
                f"Factor: {factor_col}",
            ],
            "confidence": confidence,
            "effect_size": round(cramer_v, 4),
            "adjusted_p_value": round(p_value, 6),
            "statistical_support": {
                "test": "chi2_contingency",
                "chi2": round(chi2, 3),
                "p_value": round(p_value, 6),
                "dof": int(dof),
                "cramer_v": round(cramer_v, 4),
                "defect_rates_by_level": rates[True].round(4).to_dict() if True in rates.columns else {},
            },
            "alternative_explanations": [
                f"The association between {factor_label} and defect rate "
                f"may reflect confounding with other variables (e.g. variant mix, shift).",
                "Correlation does not confirm causation. Process investigation is required.",
            ],
            "confounders": [
                c for c in ["product_variant", "shift", "batch_id", "station_id"]
                if c != factor_col and c in df.columns
            ],
        }

    # ------------------------------------------------------------------
    # Process parameter analysis
    # ------------------------------------------------------------------

    def _test_process_parameters(self, inspection_df: pd.DataFrame,
                                  process_df: pd.DataFrame,
                                  defect_family: str) -> List[Dict]:
        alpha = self._assoc_cfg.get("alpha", 0.05)
        min_effect = self._assoc_cfg.get("min_effect_size", 0.1)
        hyps = []

        # Merge inspection results with process parameters
        merge_col = "unit_id" if "unit_id" in process_df.columns else None
        if merge_col is None:
            return []

        merged = inspection_df[["unit_id", "inspection_result", "defect_family"]].merge(
            process_df, on=merge_col, how="inner"
        )
        if len(merged) < 20:
            return []

        is_defect = (merged["defect_family"] == defect_family) & \
                    (merged["inspection_result"] == "defective")

        # Identify numeric process columns (dynamically from data)
        numeric_cols = merged.select_dtypes(include=np.number).columns.tolist()
        exclude = {"unit_id", "anomaly_score", "confidence", "is_novel_defect"}
        param_cols = [c for c in numeric_cols if c not in exclude and not c.startswith("_ext_")]

        for param in param_cols:
            series = merged[param].dropna()
            if len(series) < 20:
                continue

            defect_vals = merged.loc[is_defect, param].dropna()
            normal_vals = merged.loc[~is_defect, param].dropna()

            if len(defect_vals) < 5 or len(normal_vals) < 5:
                continue

            try:
                t_stat, p_value = stats.ttest_ind(defect_vals, normal_vals)
            except Exception:
                continue

            if p_value > alpha:
                continue

            # Point-biserial correlation as effect size
            try:
                corr, _ = stats.pointbiserialr(is_defect.astype(float), merged[param].fillna(merged[param].median()))
                effect_size = abs(corr)
            except Exception:
                effect_size = 0.0

            if effect_size < min_effect:
                continue

            direction = "higher" if defect_vals.mean() > normal_vals.mean() else "lower"
            confidence = float(np.clip(effect_size * (1 - p_value) * 1.2, 0.1, 0.95))

            hyps.append({
                "rank": 0,
                "defect_family": defect_family,
                "hypothesis": (
                    f"Statistical analysis indicates that {param} is "
                    f"significantly {direction} in units with {defect_family} defects "
                    f"(defective mean: {defect_vals.mean():.2f} vs normal mean: {normal_vals.mean():.2f})."
                ),
                "evidence": [
                    f"t-test: t={t_stat:.3f}, p={p_value:.5f}",
                    f"Point-biserial correlation: r={corr:.3f}",
                    f"{defect_family} defective mean {param}: {defect_vals.mean():.3f} ± {defect_vals.std():.3f}",
                    f"Normal mean {param}: {normal_vals.mean():.3f} ± {normal_vals.std():.3f}",
                ],
                "confidence": confidence,
                "effect_size": round(float(effect_size), 4),
                "adjusted_p_value": round(float(p_value), 6),
                "statistical_support": {
                    "test": "ttest_ind",
                    "t_stat": round(float(t_stat), 3),
                    "p_value": round(float(p_value), 6),
                    "point_biserial_r": round(float(corr), 4),
                    "defect_mean": round(float(defect_vals.mean()), 3),
                    "normal_mean": round(float(normal_vals.mean()), 3),
                    "parameter": param,
                },
                "alternative_explanations": [
                    f"The {param} difference may be confounded by station, batch, or variant effects.",
                    "Temporal ordering should be verified to confirm precedence.",
                ],
                "confounders": ["station_id", "batch_id", "product_variant"],
            })

        return hyps

    # ------------------------------------------------------------------
    # Temporal trend
    # ------------------------------------------------------------------

    def _test_temporal_trend(self, inspection_df: pd.DataFrame,
                              defect_family: str) -> Optional[Dict]:
        try:
            df = inspection_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

            df["is_defect"] = ((df.get("defect_family", "") == defect_family) &
                               (df.get("inspection_result", "") == "defective")).astype(int)

            # Rolling defect rate
            df["time_index"] = np.arange(len(df))
            corr, p_value = stats.spearmanr(df["time_index"], df["is_defect"])

            if p_value > self._assoc_cfg.get("alpha", 0.05) or abs(corr) < 0.05:
                return None

            direction = "increasing" if corr > 0 else "decreasing"
            confidence = float(np.clip(abs(corr) * (1 - p_value) * 1.5, 0.05, 0.85))

            return {
                "rank": 0,
                "defect_family": defect_family,
                "hypothesis": (
                    f"Statistical analysis shows a {direction} temporal trend in "
                    f"{defect_family} defect rate (Spearman ρ={corr:.3f}). "
                    f"This may indicate process drift or cumulative equipment degradation."
                ),
                "evidence": [
                    f"Spearman correlation with time index: ρ={corr:.3f}, p={p_value:.5f}",
                    f"Trend direction: {direction}",
                ],
                "confidence": confidence,
                "effect_size": round(abs(corr), 4),
                "adjusted_p_value": round(p_value, 6),
                "statistical_support": {
                    "test": "spearmanr_temporal",
                    "spearman_rho": round(corr, 4),
                    "p_value": round(p_value, 6),
                },
                "alternative_explanations": [
                    "Trend may reflect batch-to-batch material variation rather than process drift.",
                    "Seasonal or production schedule effects may confound temporal trends.",
                ],
                "confounders": ["batch_id", "station_id", "product_variant"],
            }
        except Exception as exc:
            logger.debug("Temporal trend test failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_defect_families(self, inspection_df: pd.DataFrame) -> List[str]:
        """Get unique defect families from data (never hardcoded)."""
        if "defect_family" not in inspection_df.columns:
            return []
        families = inspection_df[
            (inspection_df.get("inspection_result", "") == "defective") &
            inspection_df["defect_family"].notna() &
            (inspection_df["defect_family"] != "") &
            (inspection_df["defect_family"] != "None")
        ]["defect_family"].unique().tolist()
        return families

    # ------------------------------------------------------------------
    # SHAP analysis (lightweight version using LightGBM)
    # ------------------------------------------------------------------

    def run_shap_analysis(self, inspection_df: pd.DataFrame,
                           process_df: pd.DataFrame,
                           defect_family: Optional[str] = None
                           ) -> Optional[Dict]:
        """
        Train LightGBM to predict defect occurrence from process parameters.
        Returns SHAP-based feature importances.
        """
        try:
            import lightgbm as lgb
            import shap
        except ImportError:
            logger.warning("LightGBM or SHAP not available. Skipping SHAP analysis.")
            return None

        shap_cfg = self._cfg["shap_analysis"]

        merge_col = "unit_id" if "unit_id" in process_df.columns else None
        if merge_col is None:
            return None

        merged = inspection_df[["unit_id", "inspection_result", "defect_family"]].merge(
            process_df, on=merge_col, how="inner"
        )

        if defect_family:
            target = ((merged["defect_family"] == defect_family) &
                      (merged["inspection_result"] == "defective")).astype(int)
        else:
            target = (merged["inspection_result"] == "defective").astype(int)

        numeric_cols = merged.select_dtypes(include=np.number).columns.tolist()
        exclude = {"unit_id", "anomaly_score", "confidence"}
        feature_cols = [c for c in numeric_cols if c not in exclude and not c.startswith("_ext_")]

        if not feature_cols or target.sum() < 10:
            return None

        X = merged[feature_cols].fillna(merged[feature_cols].median())
        y = target

        model = lgb.LGBMClassifier(
            n_estimators=shap_cfg["n_estimators"],
            max_depth=shap_cfg["max_depth"],
            learning_rate=shap_cfg["learning_rate"],
            random_state=shap_cfg["random_state"],
            verbose=-1,
        )
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_shap = np.abs(shap_values).mean(axis=0)
        top_n = shap_cfg.get("top_features", 5)
        top_idx = np.argsort(mean_shap)[::-1][:top_n]

        return {
            "defect_family": defect_family or "all",
            "feature_importances": [
                {
                    "feature": feature_cols[i],
                    "mean_shap": round(float(mean_shap[i]), 4),
                    "direction": "increases_defect" if float(np.mean(shap_values[:, i])) > 0
                                 else "decreases_defect",
                }
                for i in top_idx
            ],
            "model_accuracy": round(float((model.predict(X) == y).mean()), 4),
        }
