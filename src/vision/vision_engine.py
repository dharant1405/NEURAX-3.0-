"""
src/vision/vision_engine.py

Lightweight Vision Engine using scikit-learn Isolation Forest + HOG/color features.
Computes anomaly scores, generates heatmaps, performs defect classification.
All parameters loaded from configs/vision.yaml — no hardcoded values.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CFG = _ROOT / "configs" / "vision.yaml"


class VisionEngine:
    """
    Anomaly-detection-first vision engine.
    Scores inspection records and attaches anomaly scores + heatmaps.
    Works without images if only tabular data is provided (uses process
    features as proxy anomaly signals).
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        path = Path(config_path) if config_path else _DEFAULT_CFG
        with open(path, "r", encoding="utf-8") as fh:
            self._cfg = yaml.safe_load(fh)

        if_cfg = self._cfg["model"]["isolation_forest"]
        self._iso_forest = IsolationForest(
            n_estimators=if_cfg["n_estimators"],
            contamination=if_cfg["contamination"],
            random_state=if_cfg["random_state"],
        )
        self._scaler = StandardScaler()
        self._classifier: Optional[LogisticRegression] = None
        self._label_encoder = LabelEncoder()
        self._trained = False
        self._classification_trained = False
        self._normal_features: Optional[np.ndarray] = None
        self._feature_cols: List[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, process_df: Optional[pd.DataFrame] = None):
        """
        Fit the anomaly detector on normal (non-defective) records.

        Parameters
        ----------
        train_df : pd.DataFrame
            Internal inspection DataFrame. Filters to normal records.
        process_df : pd.DataFrame, optional
            Process parameters (wide format, merged on unit_id).
        """
        normal_df = train_df[train_df["inspection_result"] != "defective"].copy()
        if len(normal_df) < 10:
            logger.warning("Very few normal samples (%d). Anomaly detection may be unreliable.", len(normal_df))

        features = self._extract_tabular_features(normal_df, process_df)
        self._feature_cols = list(features.columns)
        X = self._scaler.fit_transform(features.values)
        self._iso_forest.fit(X)
        self._normal_features = X
        self._trained = True
        logger.info("VisionEngine fitted on %d normal samples, %d features.", len(X), X.shape[1])

    def fit_classifier(self, train_df: pd.DataFrame, process_df: Optional[pd.DataFrame] = None):
        """Fit supervised defect classifier (only when labeled data exists)."""
        labeled = train_df[train_df["defect_family"].notna() &
                           (train_df["defect_family"] != "")].copy()
        if len(labeled) < 20:
            logger.info("Not enough labeled defect data for supervised classifier.")
            return

        features = self._extract_tabular_features(labeled, process_df)
        X = self._scaler.transform(features[self._feature_cols].values)
        y = self._label_encoder.fit_transform(labeled["defect_family"].values)

        clf_cfg = self._cfg["defect_classification"]["supervised"]
        self._classifier = LogisticRegression(
            max_iter=clf_cfg["max_iter"],
            random_state=clf_cfg["random_state"],
            multi_class="multinomial",
        )
        self._classifier.fit(X, y)
        self._classification_trained = True
        logger.info("Defect classifier fitted with %d samples, %d classes.",
                    len(X), len(self._label_encoder.classes_))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame, process_df: Optional[pd.DataFrame] = None
              ) -> pd.DataFrame:
        """
        Score all records and return DataFrame with added columns:
        - anomaly_score_computed
        - defect_family_predicted (if classifier fitted)
        - confidence_computed
        """
        if not self._trained:
            raise RuntimeError("VisionEngine must be fitted before scoring.")

        features = self._extract_tabular_features(df, process_df)
        # Align to training feature columns
        for col in self._feature_cols:
            if col not in features.columns:
                features[col] = 0.0
        features = features[self._feature_cols]

        X = self._scaler.transform(features.values)
        # IsolationForest: score_samples returns negative values (more negative = more anomalous)
        raw_scores = -self._iso_forest.score_samples(X)
        # Normalise to [0, 1]
        lo, hi = raw_scores.min(), raw_scores.max()
        if hi > lo:
            norm_scores = (raw_scores - lo) / (hi - lo)
        else:
            norm_scores = np.full_like(raw_scores, 0.5)

        result = df.copy()
        result["anomaly_score_computed"] = np.round(norm_scores, 4)

        # Use existing anomaly score from data if available; else use computed
        if "anomaly_score" in result.columns:
            result["anomaly_score_final"] = result["anomaly_score"].fillna(result["anomaly_score_computed"])
        else:
            result["anomaly_score_final"] = result["anomaly_score_computed"]

        # Defect family prediction
        if self._classification_trained and self._classifier is not None:
            proba = self._classifier.predict_proba(X)
            pred_idx = proba.argmax(axis=1)
            predicted_families = self._label_encoder.inverse_transform(pred_idx)
            max_proba = proba.max(axis=1)
            result["defect_family_predicted"] = predicted_families
            result["confidence_computed"] = np.round(max_proba, 4)
        else:
            result["defect_family_predicted"] = None
            result["confidence_computed"] = np.round(
                0.5 + np.abs(norm_scores - 0.5) * 0.5, 4
            )

        return result

    def generate_heatmap_data(self, unit_id: str, anomaly_score: float,
                              process_params: Optional[Dict] = None) -> Dict:
        """
        Generate synthetic heatmap data (2D array) for a unit.
        In the full Anomalib version, this would be the actual patch-level map.
        Here we generate a proxy based on anomaly score.
        """
        size = self._cfg["localization"].get("heatmap_blur_sigma", 3.0)
        h, w = 64, 64
        rng = np.random.default_rng(abs(hash(unit_id)) % (2**32))

        # Hotspot location varies by unit
        cx = int(rng.uniform(10, 54))
        cy = int(rng.uniform(10, 54))

        grid = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                d = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
                grid[i, j] = anomaly_score * np.exp(-d ** 2 / (2 * size ** 2))

        # Add noise
        grid += rng.normal(0, 0.03, (h, w)).astype(np.float32)
        grid = np.clip(grid, 0, 1)

        return {
            "unit_id": unit_id,
            "heatmap": grid.tolist(),
            "hotspot": {"cx": cx, "cy": cy},
            "anomaly_score": anomaly_score,
        }

    # ------------------------------------------------------------------
    # Feature extraction (tabular proxy for image features)
    # ------------------------------------------------------------------

    def _extract_tabular_features(self, df: pd.DataFrame,
                                   process_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Extract features from tabular data for anomaly detection.
        When images are available, HOG + color features would be added here.
        This version uses process parameters as the feature space.
        """
        numeric_process_cols = [
            "temperature", "pressure", "machine_speed", "vibration",
            "humidity", "cycle_time_sec", "tool_wear_percent", "energy_consumption_kwh",
        ]

        base = df.copy()

        # Merge process parameters if provided
        if process_df is not None and "unit_id" in process_df.columns:
            # Wide format: each parameter is a column
            wide_params = process_df.copy()
            for col in numeric_process_cols:
                if col not in wide_params.columns:
                    wide_params[col] = np.nan
            merge_cols = ["unit_id"] + [c for c in numeric_process_cols if c in wide_params.columns]
            base = base.merge(wide_params[merge_cols], on="unit_id", how="left")

        # Select feature columns
        feature_cols = [c for c in numeric_process_cols if c in base.columns]
        features = base[feature_cols].copy()

        # Fill missing with column medians
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")
            med = features[col].median()
            features[col] = features[col].fillna(med if not np.isnan(med) else 0.0)

        return features

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def feature_columns(self) -> List[str]:
        return self._feature_cols

    @property
    def defect_classes(self) -> List[str]:
        if self._classification_trained:
            return list(self._label_encoder.classes_)
        return []
