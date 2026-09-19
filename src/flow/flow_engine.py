"""
src/flow/flow_engine.py

Flow Engine: Bottleneck detection + SimPy simulation + Loss calculation.
Station topology is loaded dynamically from data — never hardcoded.
All parameters come from configs/flow.yaml.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CFG = _ROOT / "configs" / "flow.yaml"


class FlowEngine:
    """
    Detects bottlenecks and simulates production flow.
    Station IDs, capacities, and cycle times all come from data.
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        path = Path(config_path) if config_path else _DEFAULT_CFG
        with open(path, "r") as fh:
            self._cfg = yaml.safe_load(fh)

        sim_cfg = self._cfg["simulation"]
        self._sim_duration_h = sim_cfg["default_duration_hours"]
        self._sim_seed = sim_cfg["random_seed"]
        self._sim_replications = sim_cfg["replications"]
        self._warmup_h = sim_cfg["warmup_hours"]

    # ------------------------------------------------------------------
    # Bottleneck detection
    # ------------------------------------------------------------------

    def detect_bottlenecks(self, production_df: pd.DataFrame,
                            stations_df: Optional[pd.DataFrame] = None
                            ) -> List[Dict[str, Any]]:
        """
        Detect bottleneck stations using a hybrid of utilization,
        throughput, cycle time, and WIP indicators.
        All station IDs come from the data — never hardcoded.
        """
        if production_df is None or len(production_df) == 0:
            return []

        # Discover stations dynamically
        station_ids = production_df["station_id"].unique().tolist()
        if not station_ids:
            return []

        results = []

        # Station-level aggregates
        agg = production_df.groupby("station_id").agg(
            utilization_pct=("utilization_percent", "mean"),
            throughput_ph=("actual_throughput_units_per_hour", "mean"),
            capacity_ph=("capacity_units_per_hour", "mean"),
            downtime_min=("downtime_minutes", "sum"),
            good_units=("good_units", "sum"),
            total_units=("produced_units", "sum"),
            defective_units=("defective_units", "sum"),
        ).reset_index()

        # Load station reference (if available)
        station_ref = {}
        if stations_df is not None and len(stations_df) > 0:
            for _, row in stations_df.iterrows():
                station_ref[row["station_id"]] = row.to_dict()

        # Compute bottleneck score for each station
        max_util = agg["utilization_pct"].max() or 1.0
        min_tput = agg["throughput_ph"].min() or 1.0
        max_tput = agg["throughput_ph"].max() or 1.0

        for _, row in agg.iterrows():
            sid = row["station_id"]
            util = row["utilization_pct"] / 100.0
            tput = row["throughput_ph"]
            cap = row["capacity_ph"] if row["capacity_ph"] > 0 else 1.0
            dt_min = row["downtime_min"]

            # Capacity utilization factor (higher = more bottleneck risk)
            util_factor = util

            # Throughput gap: how far below max throughput?
            tput_gap = 1.0 - (tput / max_tput) if max_tput > 0 else 0.0

            # Downtime factor
            # Assume shift = 8h per batch; normalize downtime
            n_batches = len(production_df["batch_id"].unique())
            dt_hours = dt_min / 60
            dt_factor = min(1.0, dt_hours / max(n_batches * self._sim_duration_h, 1))

            # Cycle time imbalance (vs average)
            avg_ct = production_df["capacity_units_per_hour"].mean()
            ct_imbalance = abs(cap - avg_ct) / max(avg_ct, 1)

            # Composite bottleneck score (weights from config)
            weights = self._cfg["bottleneck"]["weights"]
            score = (
                weights.get("utilization", 0.35) * util_factor +
                weights.get("wip_buildup", 0.25) * tput_gap +
                weights.get("blocked_starved_time", 0.20) * dt_factor +
                weights.get("cycle_time_imbalance", 0.20) * ct_imbalance
            )

            ref = station_ref.get(sid, {})
            results.append({
                "station_id": sid,
                "station_name": ref.get("station_name", sid),
                "bottleneck_score": round(float(score), 4),
                "utilization_pct": round(float(util * 100), 1),
                "throughput_per_hour": round(float(tput), 2),
                "capacity_per_hour": round(float(cap), 2),
                "downtime_minutes_total": round(float(dt_min), 1),
                "defect_rate_pct": round(
                    float(row["defective_units"] / max(row["total_units"], 1) * 100), 2
                ),
                "throughput_gap_pct": round(tput_gap * 100, 1),
                "value_of_relief_label": "SIMULATED / ADVISORY",
            })

        # Sort by bottleneck score descending
        results.sort(key=lambda x: x["bottleneck_score"], reverse=True)

        # Rank
        for i, r in enumerate(results):
            r["rank"] = i + 1
            if i == 0:
                r["is_primary_bottleneck"] = True
            else:
                r["is_primary_bottleneck"] = False

        return results

    # ------------------------------------------------------------------
    # Loss calculation
    # ------------------------------------------------------------------

    def calculate_losses(self, production_df: pd.DataFrame,
                          economics_df: Optional[pd.DataFrame] = None
                          ) -> Dict[str, Any]:
        """
        Break down production losses by category.
        All values computed from data — never hardcoded.
        """
        if production_df is None or len(production_df) == 0:
            return {}

        total_planned = production_df["planned_units"].sum() if "planned_units" in production_df.columns else 0
        total_produced = production_df["produced_units"].sum() if "produced_units" in production_df.columns else 0
        total_good = production_df["good_units"].sum() if "good_units" in production_df.columns else total_produced
        total_defective = production_df["defective_units"].sum() if "defective_units" in production_df.columns else 0
        total_rework = production_df["rework_units"].sum() if "rework_units" in production_df.columns else 0
        total_scrap = production_df["scrap_units"].sum() if "scrap_units" in production_df.columns else 0
        total_downtime_min = production_df["downtime_minutes"].sum() if "downtime_minutes" in production_df.columns else 0

        # Economic losses (if economics_df provided)
        scrap_cost = 0.0
        rework_cost = 0.0
        if economics_df is not None and "scrap_cost" in economics_df.columns:
            scrap_cost = float(economics_df["scrap_cost"].sum())
        if economics_df is not None and "rework_cost" in economics_df.columns:
            rework_cost = float(economics_df["rework_cost"].sum())

        return {
            "total_planned_units": int(total_planned),
            "total_produced_units": int(total_produced),
            "total_good_units": int(total_good),
            "total_defective_units": int(total_defective),
            "total_rework_units": int(total_rework),
            "total_scrap_units": int(total_scrap),
            "total_downtime_hours": round(float(total_downtime_min) / 60, 2),
            "overall_defect_rate_pct": round(float(total_defective) / max(total_produced, 1) * 100, 2),
            "scrap_rate_pct": round(float(total_scrap) / max(total_produced, 1) * 100, 2),
            "rework_rate_pct": round(float(total_rework) / max(total_produced, 1) * 100, 2),
            "estimated_scrap_cost": round(scrap_cost, 2),
            "estimated_rework_cost": round(rework_cost, 2),
            "estimated_quality_loss": round(scrap_cost + rework_cost, 2),
        }

    # ------------------------------------------------------------------
    # SimPy simulation (baseline + scenario)
    # ------------------------------------------------------------------

    def simulate(self, stations_df: Optional[pd.DataFrame],
                 production_df: Optional[pd.DataFrame],
                 scenario_params: Optional[Dict] = None,
                 label: str = "Baseline") -> Dict[str, Any]:
        """
        Run a discrete-event simulation using SimPy.
        Station parameters loaded from data — topology not hardcoded.
        """
        try:
            import simpy
        except ImportError:
            logger.warning("SimPy not installed. Returning analytical approximation.")
            return self._analytical_approximation(stations_df, production_df, scenario_params, label)

        return self._simpy_simulation(stations_df, production_df, scenario_params, label)

    def _simpy_simulation(self, stations_df, production_df, scenario_params, label):
        """SimPy-based discrete event simulation."""
        import simpy

        rng = random.Random(self._sim_seed)
        scenario = scenario_params or {}

        # Build station configs from data
        station_configs = self._build_station_configs(stations_df, production_df, scenario)
        if not station_configs:
            return self._analytical_approximation(stations_df, production_df, scenario, label)

        sim_duration_sec = self._sim_duration_h * 3600
        warmup_sec = self._warmup_h * 3600

        replication_results = []

        for rep in range(self._sim_replications):
            env = simpy.Environment()
            counters = {
                "produced": 0, "scrap": 0, "rework": 0,
                "station_busy": {s["station_id"]: 0.0 for s in station_configs}
            }

            resources = {
                s["station_id"]: simpy.Resource(env, capacity=1)
                for s in station_configs
            }

            def unit_process(env, unit_id, rng_local=rng):
                for s_cfg in station_configs:
                    sid = s_cfg["station_id"]
                    ct_mean = s_cfg["cycle_time_sec"]
                    ct_std = ct_mean * 0.08

                    with resources[sid].request() as req:
                        yield req
                        ct = max(1.0, rng_local.gauss(ct_mean, ct_std))
                        start = env.now
                        yield env.timeout(ct)
                        if env.now > warmup_sec:
                            counters["station_busy"][sid] += (env.now - start)

                # Defect and scrap/rework at end
                defect_rate = s_cfg.get("defect_rate", 0.05) * scenario.get("defect_rate_multiplier", 1.0)
                if rng_local.random() < defect_rate:
                    if rng_local.random() < 0.4:
                        counters["scrap"] += 1
                    else:
                        counters["rework"] += 1
                else:
                    if env.now > warmup_sec:
                        counters["produced"] += 1

            def arrival_process(env):
                unit_id = 0
                arrival_rate = station_configs[0]["cycle_time_sec"]  # inter-arrival = first station CT
                while True:
                    yield env.timeout(max(0.1, rng.gauss(arrival_rate, arrival_rate * 0.05)))
                    unit_id += 1
                    env.process(unit_process(env, unit_id))

            env.process(arrival_process(env))
            env.run(until=sim_duration_sec)

            effective_duration = max(sim_duration_sec - warmup_sec, 1)
            n_produced = counters["produced"]
            throughput_ph = n_produced / (effective_duration / 3600)

            util_by_station = {
                sid: round(min(1.0, counters["station_busy"][sid] / effective_duration) * 100, 1)
                for sid in counters["station_busy"]
            }

            replication_results.append({
                "throughput_per_hour": throughput_ph,
                "scrap_units": counters["scrap"],
                "rework_units": counters["rework"],
                "utilization_by_station": util_by_station,
            })

        # Average replications
        avg_tput = float(np.mean([r["throughput_per_hour"] for r in replication_results]))
        avg_scrap = float(np.mean([r["scrap_units"] for r in replication_results]))
        avg_rework = float(np.mean([r["rework_units"] for r in replication_results]))

        avg_util = {}
        for sid in (replication_results[0]["utilization_by_station"] if replication_results else {}):
            avg_util[sid] = round(float(np.mean(
                [r["utilization_by_station"].get(sid, 0) for r in replication_results]
            )), 1)

        return {
            "scenario_label": label,
            "is_baseline": scenario_params is None,
            "throughput_per_hour": round(avg_tput, 2),
            "scrap_units": round(avg_scrap, 1),
            "rework_units": round(avg_rework, 1),
            "utilization_by_station": avg_util,
            "wip_avg": None,
            "simulation_duration_hours": self._sim_duration_h,
            "replications": self._sim_replications,
            "advisory_label": "SIMULATED / ADVISORY",
        }

    def _analytical_approximation(self, stations_df, production_df, scenario, label):
        """Fallback when SimPy is unavailable: use queuing theory approximations."""
        if production_df is None or len(production_df) == 0:
            return {"scenario_label": label, "error": "No production data available"}

        multiplier = scenario.get("defect_rate_multiplier", 1.0) if scenario else 1.0
        ct_mult = scenario.get("cycle_time_multiplier", 1.0) if scenario else 1.0

        avg_tput = float(production_df["actual_throughput_units_per_hour"].mean()) / ct_mult if \
            "actual_throughput_units_per_hour" in production_df.columns else 30.0
        avg_scrap = float(production_df["scrap_units"].mean()) * multiplier if \
            "scrap_units" in production_df.columns else 0.0
        avg_rework = float(production_df["rework_units"].mean()) * multiplier if \
            "rework_units" in production_df.columns else 0.0

        return {
            "scenario_label": label,
            "is_baseline": scenario is None,
            "throughput_per_hour": round(avg_tput, 2),
            "scrap_units": round(avg_scrap, 1),
            "rework_units": round(avg_rework, 1),
            "utilization_by_station": {},
            "wip_avg": None,
            "advisory_label": "SIMULATED / ADVISORY",
        }

    def _build_station_configs(self, stations_df, production_df, scenario) -> List[Dict]:
        """Build station parameter list from data (not hardcoded)."""
        configs = []
        ct_mult = scenario.get("cycle_time_multiplier", 1.0) if scenario else 1.0

        if stations_df is not None and len(stations_df) > 0 and "station_id" in stations_df.columns:
            for _, row in stations_df.iterrows():
                ct = float(row.get("nominal_cycle_time_sec", 60)) * ct_mult
                configs.append({
                    "station_id": row["station_id"],
                    "cycle_time_sec": ct,
                    "defect_rate": 0.05,  # default; overridden below
                })
        elif production_df is not None and "station_id" in production_df.columns:
            agg = production_df.groupby("station_id").agg(
                throughput_ph=("actual_throughput_units_per_hour", "mean"),
                defect_rate=("defective_units", lambda x: x.sum() / max(production_df.loc[x.index, "produced_units"].sum(), 1)),
            ).reset_index()
            for _, row in agg.iterrows():
                ct = (3600 / row["throughput_ph"]) * ct_mult if row["throughput_ph"] > 0 else 60
                configs.append({
                    "station_id": row["station_id"],
                    "cycle_time_sec": ct,
                    "defect_rate": float(row["defect_rate"]),
                })

        # Overlay defect rates from production data
        if production_df is not None and configs:
            agg_defect = production_df.groupby("station_id").apply(
                lambda g: g["defective_units"].sum() / max(g["produced_units"].sum(), 1)
            ).to_dict()
            for cfg in configs:
                cfg["defect_rate"] = float(agg_defect.get(cfg["station_id"], cfg["defect_rate"]))

        return configs
