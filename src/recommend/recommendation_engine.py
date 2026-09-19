"""
src/recommend/recommendation_engine.py

Recommendation Engine: Generates evidence-based, advisory recommendations
by combining root-cause hypotheses + bottleneck findings + economics.
All recommendations are explicitly labelled SIMULATED / ADVISORY.
Language is advisory — never imperative.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CFG = _ROOT / "configs" / "simulation.yaml"

ADVISORY_LABEL = "SIMULATED / ADVISORY"


class RecommendationEngine:
    """
    Ranks and generates advisory recommendations from multiple evidence sources.
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        path = Path(config_path) if config_path else _DEFAULT_CFG
        with open(path, "r") as fh:
            self._cfg = yaml.safe_load(fh)

    def generate(self,
                 hypotheses: List[Dict],
                 bottleneck_results: List[Dict],
                 economics_summary: Dict,
                 scenario_delta: Optional[Dict] = None
                 ) -> List[Dict[str, Any]]:
        """
        Generate ranked advisory recommendations from engine outputs.

        All recommendation text is advisory ("Simulation indicates...")
        Expected impacts are labelled SIMULATED / ADVISORY.
        """
        recommendations: List[Dict] = []

        # ── From root-cause hypotheses ─────────────────────────────────
        for hyp in hypotheses[:5]:
            rec = self._rec_from_hypothesis(hyp, economics_summary)
            if rec:
                recommendations.append(rec)

        # ── From bottleneck results ────────────────────────────────────
        for bn in bottleneck_results[:3]:
            rec = self._rec_from_bottleneck(bn, economics_summary)
            if rec:
                recommendations.append(rec)

        # ── Sort by confidence × estimated impact ─────────────────────
        def score_rec(r):
            conf = r.get("confidence", 0.5)
            impact = r.get("expected_impact", {}).get("raw_score", 0)
            return conf * (1 + impact)

        recommendations.sort(key=score_rec, reverse=True)

        # Rank
        for i, rec in enumerate(recommendations):
            rec["rank"] = i + 1
            rec["status"] = ADVISORY_LABEL

        return recommendations

    # ------------------------------------------------------------------
    # From root-cause hypothesis
    # ------------------------------------------------------------------

    def _rec_from_hypothesis(self, hyp: Dict, economics_summary: Dict) -> Optional[Dict]:
        confidence = hyp.get("confidence", 0.0)
        if confidence < 0.15:
            return None

        defect_family = hyp.get("defect_family", "unknown defect type")
        stat_support = hyp.get("statistical_support", {})
        effect_size = hyp.get("effect_size", 0.0)

        # Extract the most informative piece of evidence
        evidence = hyp.get("evidence", [])
        primary_factor = self._extract_factor(stat_support, evidence)

        # Language: advisory, not imperative
        action = (
            f"Simulation indicates that investigating and addressing the "
            f"conditions associated with elevated {defect_family} defect rates "
            f"(evidence: {primary_factor}) could, under current simulation assumptions, "
            f"reduce the {defect_family} defect rate."
        )

        # Estimated impact (all labelled advisory)
        total_profit = economics_summary.get("summary", {}).get("total_profit", 0)
        n_units = max(economics_summary.get("summary", {}).get("n_units", 1), 1)
        profit_per_unit = total_profit / n_units

        # Conservative estimate: reduce defect-related losses by confidence * effect_size fraction
        estimated_margin_gain = profit_per_unit * confidence * effect_size * 0.5

        return {
            "rank": 0,
            "action": action,
            "evidence": evidence[:3],
            "expected_impact": {
                "estimated_margin_gain_per_unit": round(estimated_margin_gain, 2),
                "raw_score": float(confidence * effect_size),
                "description": (
                    f"Simulated margin improvement of up to "
                    f"{economics_summary.get('summary', {}).get('currency', '$')}"
                    f"{abs(estimated_margin_gain):.2f} per unit "
                    f"(confidence: {confidence:.0%}, effect size: {effect_size:.3f}). "
                    f"All estimates are advisory."
                ),
            },
            "confidence": round(confidence, 4),
            "assumptions": [
                "Impact estimate assumes defect rate reduction is proportional to evidence strength.",
                "Margin calculation uses historical per-unit economics.",
                "Process change outcomes may differ from simulation predictions.",
            ],
            "alternative_explanations": hyp.get("alternative_explanations", []),
            "linked_hypothesis_rank": hyp.get("rank"),
            "status": ADVISORY_LABEL,
        }

    # ------------------------------------------------------------------
    # From bottleneck result
    # ------------------------------------------------------------------

    def _rec_from_bottleneck(self, bn: Dict, economics_summary: Dict) -> Optional[Dict]:
        score = bn.get("bottleneck_score", 0)
        if score < 0.2:
            return None

        station_id = bn.get("station_id", "unknown station")
        station_name = bn.get("station_name", station_id)
        util = bn.get("utilization_pct", 0)
        tput_gap = bn.get("throughput_gap_pct", 0)

        action = (
            f"Simulation indicates that relieving the production constraint at "
            f"{station_name} (bottleneck score: {score:.2f}, utilization: {util:.0f}%) "
            f"could, under current assumptions, increase simulated system throughput. "
            f"Possible levers include: reducing cycle time, reducing downtime, "
            f"or redistributing workload."
        )

        total_profit = economics_summary.get("summary", {}).get("total_profit", 0)
        n_units = max(economics_summary.get("summary", {}).get("n_units", 1), 1)
        profit_per_unit = total_profit / n_units
        estimated_gain = profit_per_unit * tput_gap / 100 * 8  # per shift

        return {
            "rank": 0,
            "action": action,
            "evidence": [
                f"Bottleneck score: {score:.3f} (rank {bn.get('rank', '?')})",
                f"Utilization: {util:.1f}%",
                f"Throughput gap vs line maximum: {tput_gap:.1f}%",
                f"Downtime at this station: {bn.get('downtime_minutes_total', 0):.0f} min",
            ],
            "expected_impact": {
                "estimated_margin_gain_per_shift": round(estimated_gain, 2),
                "raw_score": float(score),
                "description": (
                    f"Simulated margin improvement of approximately "
                    f"{economics_summary.get('summary', {}).get('currency', '$')}"
                    f"{abs(estimated_gain):.2f} per shift if throughput gap at "
                    f"{station_name} is closed. Advisory only."
                ),
            },
            "confidence": round(min(score, 0.90), 4),
            "assumptions": [
                "Throughput gain assumes no other station becomes the new bottleneck.",
                "Margin gain estimated from historical per-unit economics.",
                "Downtime reduction feasibility not validated.",
            ],
            "alternative_explanations": [
                f"Other stations may limit throughput after {station_name} is improved.",
                "Capacity increase may require capital investment not modelled here.",
            ],
            "status": ADVISORY_LABEL,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_factor(stat_support: Dict, evidence: List[str]) -> str:
        if "parameter" in stat_support:
            return f"process parameter '{stat_support['parameter']}'"
        if evidence:
            return evidence[0][:80]
        return "see evidence panel"
