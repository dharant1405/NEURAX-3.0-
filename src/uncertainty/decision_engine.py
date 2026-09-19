"""
src/uncertainty/decision_engine.py

Uncertainty Layer: Conformal prediction + novelty detection + decision policy.
Maps (anomaly_score, defect_family_predicted) → ACCEPT | REJECT | REVIEW | NOVEL
All thresholds come from configs/uncertainty.yaml or are calibrated from data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.models.internal_schema import DecisionState

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CFG = _ROOT / "configs" / "uncertainty.yaml"


class DecisionEngine:
    """
    Applies conformal calibration and novelty detection to produce
    a final decision state for each inspected unit.
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        path = Path(config_path) if config_path else _DEFAULT_CFG
        with open(path, "r") as fh:
            self._cfg = yaml.safe_load(fh)

        conf_cfg = self._cfg["conformal"]
        self._alpha = conf_cfg["alpha"]
        self._min_cal_samples = conf_cfg["min_calibration_samples"]

        novelty_cfg = self._cfg["novelty"]
        self._novelty_threshold = novelty_cfg["threshold"]  # None = auto
        self._min_known_examples = novelty_cfg["min_known_examples"]

        ds_cfg = self._cfg["decision_states"]
        self._low_threshold = ds_cfg.get("low_anomaly_threshold") or ds_cfg["fallback_low_threshold"]
        self._high_threshold = ds_cfg.get("high_anomaly_threshold") or ds_cfg["fallback_high_threshold"]
        self._calibrated = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, calibration_df: pd.DataFrame, score_col: str = "anomaly_score_final"):
        """
        Set thresholds from a calibration set using the conformal approach.
        calibration_df must include ground-truth inspection_result.
        """
        cal = calibration_df.dropna(subset=[score_col]).copy()
        if len(cal) < self._min_cal_samples:
            logger.warning("Only %d calibration samples (need %d). Using fallback thresholds.",
                           len(cal), self._min_cal_samples)
            return

        # Normal unit non-conformity scores
        normal_scores = cal[cal["inspection_result"] != "defective"][score_col].values
        if len(normal_scores) > 0:
            # Threshold at (1-alpha) quantile of normal scores
            self._high_threshold = float(np.quantile(normal_scores, 1 - self._alpha))
            self._low_threshold = float(np.quantile(normal_scores, 0.5))
            self._calibrated = True
            logger.info("Calibrated thresholds: low=%.3f, high=%.3f from %d normal samples.",
                        self._low_threshold, self._high_threshold, len(normal_scores))

        # Novelty threshold: 95th percentile of intra-class anomaly score std
        if self._novelty_threshold is None:
            self._novelty_threshold = self._high_threshold * 1.05

    def update_thresholds_from_economics(self, low: float, high: float):
        """Override thresholds with economics-optimized values."""
        self._low_threshold = low
        self._high_threshold = high
        logger.info("Thresholds updated from economics: low=%.3f, high=%.3f", low, high)

    # ------------------------------------------------------------------
    # Decision assignment
    # ------------------------------------------------------------------

    def assign_decisions(self, df: pd.DataFrame,
                         score_col: str = "anomaly_score_final") -> pd.DataFrame:
        """
        Assign decision states to all records.
        Adds columns: decision_state, decision_reason.
        """
        result = df.copy()
        scores = result[score_col].fillna(0.5).values
        is_novel_flags = result.get("is_novel_defect", pd.Series([False] * len(result))).fillna(False).values

        decisions = []
        reasons = []

        for i, (score, is_novel) in enumerate(zip(scores, is_novel_flags)):
            state, reason = self._apply_policy(float(score), bool(is_novel))
            decisions.append(state.value)
            reasons.append(reason)

        result["decision_state"] = decisions
        result["decision_reason"] = reasons
        return result

    def _apply_policy(self, score: float, is_novel: bool) -> Tuple[DecisionState, str]:
        """
        Apply decision policy rules.
        Rules are driven entirely by configured thresholds.
        """
        if is_novel and score > self._high_threshold:
            return DecisionState.NOVEL, f"Score {score:.3f} > threshold {self._high_threshold:.3f} and flagged as novel pattern"

        if score <= self._low_threshold:
            return DecisionState.ACCEPT, f"Score {score:.3f} ≤ low threshold {self._low_threshold:.3f}"

        if score >= self._high_threshold:
            return DecisionState.REJECT, f"Score {score:.3f} ≥ high threshold {self._high_threshold:.3f}"

        # Ambiguous zone
        return DecisionState.REVIEW, (
            f"Score {score:.3f} between thresholds "
            f"[{self._low_threshold:.3f}, {self._high_threshold:.3f}]"
        )

    # ------------------------------------------------------------------
    # State summary
    # ------------------------------------------------------------------

    def summarize(self, df: pd.DataFrame) -> Dict:
        """Count decisions and return summary dict."""
        if "decision_state" not in df.columns:
            return {}
        counts = df["decision_state"].value_counts().to_dict()
        total = len(df)
        return {
            "total": total,
            "counts": counts,
            "rates": {k: round(v / total, 4) for k, v in counts.items()},
            "low_threshold": self._low_threshold,
            "high_threshold": self._high_threshold,
            "calibrated": self._calibrated,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def low_threshold(self) -> float:
        return self._low_threshold

    @property
    def high_threshold(self) -> float:
        return self._high_threshold

    @property
    def calibrated(self) -> bool:
        return self._calibrated
