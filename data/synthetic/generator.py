"""
data/synthetic/generator.py

Generates a realistic synthetic manufacturing dataset.
All parameters are configurable — no hardcoded business values.
Outputs CSV files to data/synthetic/ directory.

Run:
    python data/synthetic/generator.py
    python data/synthetic/generator.py --output-dir data/synthetic --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Generator Configuration (all values are overridable via CLI or code)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "seed": 42,
    "n_units": 12000,
    "n_stations": 5,
    "n_variants": 4,
    "n_batches": 20,
    "n_shifts": 3,
    "n_operators": 12,
    "start_date": "2024-01-01",
    "stations": [
        {"id": "S1", "name": "Raw Material Prep",     "process": "Preparation",  "cycle_sec": 45,  "capacity_ph": 80, "temp_min": 20, "temp_max": 30, "pres_min": 1.0, "pres_max": 2.0},
        {"id": "S2", "name": "Machining",              "process": "Machining",    "cycle_sec": 72,  "capacity_ph": 50, "temp_min": 25, "temp_max": 40, "pres_min": 2.0, "pres_max": 4.0},
        {"id": "S3", "name": "Thermal Treatment",      "process": "Thermal",      "cycle_sec": 90,  "capacity_ph": 40, "temp_min": 150, "temp_max": 200, "pres_min": 1.5, "pres_max": 3.0},
        {"id": "S4", "name": "Assembly & Integration", "process": "Assembly",     "cycle_sec": 120, "capacity_ph": 30, "temp_min": 20, "temp_max": 35, "pres_min": 1.0, "pres_max": 2.5},
        {"id": "S5", "name": "Final Inspection",       "process": "QC",           "cycle_sec": 30,  "capacity_ph": 120, "temp_min": 18, "temp_max": 28, "pres_min": 1.0, "pres_max": 1.5},
    ],
    "variants": [
        {"id": "VA", "name": "Variant Alpha", "base_defect_rate": 0.04, "price": 850,  "material": 180},
        {"id": "VB", "name": "Variant Beta",  "base_defect_rate": 0.06, "price": 620,  "material": 140},
        {"id": "VC", "name": "Variant Gamma", "base_defect_rate": 0.03, "price": 1200, "material": 250},
        {"id": "VD", "name": "Variant Delta", "base_defect_rate": 0.08, "price": 490,  "material": 110},
    ],
    "defect_types": [
        {"type": "Scratch",       "family": "Surface_Defect",    "station": "S2", "cause": "High vibration at machining station; tool wear", "severity": "minor",  "resolution": "rework",  "cost": 60},
        {"type": "Crack",         "family": "Structural_Defect", "station": "S3", "cause": "Temperature spike during thermal treatment",       "severity": "major",  "resolution": "scrap",   "cost": 250},
        {"type": "Dent",          "family": "Surface_Defect",    "station": "S4", "cause": "Assembly impact; misaligned tooling",               "severity": "minor",  "resolution": "rework",  "cost": 45},
        {"type": "Discoloration", "family": "Thermal_Defect",    "station": "S3", "cause": "Temperature exceedance; wrong atmosphere settings",  "severity": "minor",  "resolution": "rework",  "cost": 35},
        {"type": "Misalignment",  "family": "Assembly_Defect",   "station": "S4", "cause": "Fixture wear; operator error; material variation",  "severity": "major",  "resolution": "rework",  "cost": 80},
        {"type": "Surface_Pit",   "family": "Surface_Defect",    "station": "S2", "cause": "Coolant contamination; tool chipping",              "severity": "minor",  "resolution": "rework",  "cost": 50},
        {"type": "Burr",          "family": "Machining_Defect",  "station": "S2", "cause": "Dull tooling; excessive feed rate",                 "severity": "minor",  "resolution": "rework",  "cost": 30},
        {"type": "Porosity",      "family": "Material_Defect",   "station": "S1", "cause": "Raw material batch defect; supplier quality issue", "severity": "major",  "resolution": "scrap",   "cost": 200},
        {"type": "Delamination",  "family": "Structural_Defect", "station": "S3", "cause": "Insufficient bonding time; contamination",         "severity": "critical","resolution": "scrap",   "cost": 300},
        {"type": "None",          "family": "None",              "station": None, "cause": "No defect",                                         "severity": "none",   "resolution": "none",    "cost": 0},
    ],
    "novel_defect_rate": 0.008,   # fraction of units with truly novel defects
    "economics": {
        "processing_cost_base": 80,
        "energy_cost_base": 15,
        "labor_cost_base": 40,
        "inspection_cost": 5,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Root-Cause Patterns (injected into synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

PATTERNS = {
    # Pattern 1: S3 temperature drift (batches B013–B020)
    "s3_temp_drift": {
        "station": "S3",
        "param": "temperature",
        "start_batch_idx": 12,    # 0-indexed; batch 13 onwards
        "drift_per_batch": 3.5,   # degrees increase per batch
        "affected_defects": ["Crack", "Discoloration"],
        "defect_rate_multiplier": 1.8,
    },
    # Pattern 2: S2 vibration increases due to tool wear (batches B008–B015)
    "s2_vibration_drift": {
        "station": "S2",
        "param": "vibration",
        "start_batch_idx": 7,
        "drift_per_batch": 0.12,
        "affected_defects": ["Scratch", "Surface_Pit"],
        "defect_rate_multiplier": 1.5,
    },
    # Pattern 3: S4 is the bottleneck (cycle time is longest, capacity lowest)
    # Encoded in station config above (S4: 120s, 30 units/hour)

    # Pattern 4: Material-related problem in batches B009–B011
    "material_batch_issue": {
        "batches": [8, 9, 10],   # 0-indexed
        "station": "S1",
        "affected_defects": ["Porosity"],
        "defect_rate_multiplier": 4.0,
        "extra_scrap_rate": 0.06,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Generator Class
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Generates a complete, internally consistent synthetic manufacturing dataset.
    All relationships are deliberate (not purely random) to support AI analysis.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.cfg = config or DEFAULT_CONFIG
        self._rng = np.random.default_rng(self.cfg["seed"])
        random.seed(self.cfg["seed"])

        self._stations = self.cfg["stations"]
        self._variants = self.cfg["variants"]
        self._defect_types = self.cfg["defect_types"]
        self._n_units = self.cfg["n_units"]
        self._n_batches = self.cfg["n_batches"]
        self._n_shifts = self.cfg["n_shifts"]
        self._n_operators = self.cfg["n_operators"]
        self._start_date = datetime.fromisoformat(self.cfg["start_date"])

        # Index helpers
        self._station_ids = [s["id"] for s in self._stations]
        self._variant_ids = [v["id"] for v in self._variants]
        self._batch_ids = [f"B{i+1:03d}" for i in range(self._n_batches)]
        self._shift_names = [f"Shift_{i+1}" for i in range(self._n_shifts)]
        self._operator_ids = [f"OP{i+1:03d}" for i in range(self._n_operators)]

    # ------------------------------------------------------------------
    # Generate all tables
    # ------------------------------------------------------------------

    def generate_all(self, output_dir: str | Path) -> Dict[str, pd.DataFrame]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating synthetic dataset: %d units, %d stations, %d variants, %d batches",
                    self._n_units, len(self._stations), len(self._variants), self._n_batches)

        tables = {}

        # Generate in dependency order
        tables["stations"] = self._gen_stations()
        tables["defect_catalog"] = self._gen_defect_catalog()
        tables["batches"] = self._gen_batches()  # placeholder; updated at end
        tables["products"], tables["inspection"], tables["process_parameters"] = \
            self._gen_unit_level_data()
        tables["production"] = self._gen_production_aggregates(
            tables["products"], tables["inspection"], tables["process_parameters"]
        )
        tables["downtime"] = self._gen_downtime_events(tables["production"])
        tables["economics"] = self._gen_economics(tables["products"], tables["inspection"])
        # Recompute batches with actual stats
        tables["batches"] = self._summarize_batches(
            tables["products"], tables["inspection"], tables["process_parameters"]
        )

        # Write to disk
        for name, df in tables.items():
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            logger.info("Wrote %s (%d rows)", path.name, len(df))

        return tables

    # ------------------------------------------------------------------
    # Stations
    # ------------------------------------------------------------------

    def _gen_stations(self) -> pd.DataFrame:
        rows = []
        for s in self._stations:
            rows.append({
                "station_id": s["id"],
                "station_name": s["name"],
                "process_type": s["process"],
                "nominal_cycle_time_sec": s["cycle_sec"],
                "maximum_capacity_per_hour": s["capacity_ph"],
                "normal_temperature_min": s["temp_min"],
                "normal_temperature_max": s["temp_max"],
                "normal_pressure_min": s["pres_min"],
                "normal_pressure_max": s["pres_max"],
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Defect catalog
    # ------------------------------------------------------------------

    def _gen_defect_catalog(self) -> pd.DataFrame:
        rows = []
        for d in self._defect_types:
            rows.append({
                "defect_type": d["type"],
                "defect_family": d["family"],
                "typical_station": d.get("station"),
                "possible_causes": d.get("cause"),
                "severity": d.get("severity"),
                "scrap_or_rework": d.get("resolution"),
                "typical_cost": d.get("cost"),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Batch stubs (fully computed later)
    # ------------------------------------------------------------------

    def _gen_batches(self) -> pd.DataFrame:
        rows = []
        batch_start = self._start_date
        for i, bid in enumerate(self._batch_ids):
            variant = self._variants[i % len(self._variants)]
            duration_h = self._rng.uniform(8, 24)
            batch_end = batch_start + timedelta(hours=duration_h)
            qty = int(self._rng.integers(400, 800))
            rows.append({
                "batch_id": bid,
                "product_variant": variant["id"],
                "material_batch": f"MAT-{bid}",
                "start_time": batch_start.isoformat(),
                "end_time": batch_end.isoformat(),
                "planned_quantity": qty,
                "actual_quantity": qty,
                "defect_rate_percent": 0.0,
                "rework_rate_percent": 0.0,
                "scrap_rate_percent": 0.0,
                "average_cycle_time_sec": 0.0,
                "average_temperature": 0.0,
                "average_pressure": 0.0,
                "average_machine_speed": 0.0,
                "process_drift_score": 0.0,
            })
            batch_start = batch_end + timedelta(hours=self._rng.uniform(0.5, 2))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Unit-level: products, inspection, process_parameters
    # ------------------------------------------------------------------

    def _gen_unit_level_data(self):
        products_rows = []
        inspection_rows = []
        process_rows = []

        # Assign units to batches proportionally
        units_per_batch = self._n_units // self._n_batches
        remainder = self._n_units % self._n_batches

        unit_counter = 1
        ts = self._start_date

        for batch_idx, batch_id in enumerate(self._batch_ids):
            variant = self._variants[batch_idx % len(self._variants)]
            n_units_this_batch = units_per_batch + (1 if batch_idx < remainder else 0)
            batch_ts = ts

            for u in range(n_units_this_batch):
                unit_id = f"U{unit_counter:06d}"
                unit_counter += 1

                # Each unit passes through all stations in sequence
                # The inspection station is the last station
                last_station = self._stations[-1]
                primary_station = self._stations[batch_idx % (len(self._stations) - 1)]

                shift = self._shift_names[u % len(self._shift_names)]
                operator = self._operator_ids[u % len(self._operator_ids)]
                production_ts = batch_ts + timedelta(
                    seconds=u * primary_station["cycle_sec"] + self._rng.uniform(-5, 5)
                )

                products_rows.append({
                    "product_id": unit_id,
                    "product_variant": variant["id"],
                    "batch_id": batch_id,
                    "production_timestamp": production_ts.isoformat(),
                    "station_id": primary_station["id"],
                    "shift": shift,
                    "operator_id": operator,
                })

                # Process parameters for primary station
                proc = self._gen_process_params(unit_id, batch_id, primary_station, batch_idx, variant)
                process_rows.append(proc)

                # Determine defect based on process conditions + patterns
                defect_info, is_novel = self._determine_defect(
                    variant, primary_station, batch_idx, proc
                )

                # Anomaly score: correlated with defect presence + process deviation
                base_score = self._compute_anomaly_score(defect_info, proc, primary_station, batch_idx)
                confidence = float(np.clip(self._rng.normal(0.82, 0.10), 0.40, 0.99))

                inspection_ts = production_ts + timedelta(
                    seconds=last_station["cycle_sec"] + self._rng.uniform(-3, 3)
                )

                inspection_result = "defective" if defect_info["type"] != "None" else "good"
                if is_novel:
                    inspection_result = "defective"

                inspection_rows.append({
                    "product_id": unit_id,
                    "inspection_timestamp": inspection_ts.isoformat(),
                    "inspection_result": inspection_result,
                    "defect_type": "Novel_Defect" if is_novel else defect_info["type"],
                    "defect_family": "Unknown" if is_novel else defect_info["family"],
                    "defect_location": self._random_location() if inspection_result == "defective" else "",
                    "defect_severity": "critical" if is_novel else defect_info.get("severity", "none"),
                    "inspection_confidence": round(confidence, 4),
                    "anomaly_score": round(base_score, 4),
                    "is_novel_defect": is_novel,
                })

            ts = batch_ts + timedelta(
                seconds=n_units_this_batch * primary_station["cycle_sec"] / 3600 * 3600
            )

        return (
            pd.DataFrame(products_rows),
            pd.DataFrame(inspection_rows),
            pd.DataFrame(process_rows),
        )

    # ------------------------------------------------------------------
    # Process parameter generation
    # ------------------------------------------------------------------

    def _gen_process_params(self, unit_id, batch_id, station, batch_idx, variant) -> dict:
        s = station
        sid = s["id"]

        # Base temperature with variant-specific offset
        variant_temp_offset = {"VA": 0, "VB": -5, "VC": +8, "VD": -3}.get(variant["id"], 0)
        base_temp = (s["temp_min"] + s["temp_max"]) / 2 + variant_temp_offset

        # Pattern 1: S3 temperature drift
        temp_drift = 0.0
        if sid == "S3" and batch_idx >= PATTERNS["s3_temp_drift"]["start_batch_idx"]:
            drift_batches = batch_idx - PATTERNS["s3_temp_drift"]["start_batch_idx"]
            temp_drift = drift_batches * PATTERNS["s3_temp_drift"]["drift_per_batch"]

        temperature = float(np.clip(
            self._rng.normal(base_temp + temp_drift, (s["temp_max"] - s["temp_min"]) * 0.08),
            s["temp_min"] - 5, s["temp_max"] + temp_drift + 10
        ))

        base_pressure = (s["pres_min"] + s["pres_max"]) / 2
        pressure = float(np.clip(
            self._rng.normal(base_pressure, (s["pres_max"] - s["pres_min"]) * 0.1),
            s["pres_min"] * 0.8, s["pres_max"] * 1.2
        ))

        # Base machine speed
        base_speed = 100.0  # RPM equivalent
        speed = float(np.clip(self._rng.normal(base_speed, 8), 70, 140))

        # Pattern 2: S2 vibration drift
        base_vibration = 1.5  # mm/s
        vibration_drift = 0.0
        if sid == "S2" and batch_idx >= PATTERNS["s2_vibration_drift"]["start_batch_idx"]:
            drift_batches = batch_idx - PATTERNS["s2_vibration_drift"]["start_batch_idx"]
            vibration_drift = drift_batches * PATTERNS["s2_vibration_drift"]["drift_per_batch"]
        vibration = float(np.clip(
            self._rng.normal(base_vibration + vibration_drift, 0.15),
            0.5, 6.0
        ))

        # Tool wear: increases through batch, resets at new batch
        tool_wear = float(np.clip(
            self._rng.normal(50 + batch_idx * 1.2, 8), 10, 95
        ))

        # Cycle time: increases with tool wear and temp deviation
        nominal_ct = s["cycle_sec"]
        ct_factor = 1.0 + (tool_wear - 50) * 0.002 + abs(temp_drift) * 0.005
        cycle_time = float(np.clip(
            self._rng.normal(nominal_ct * ct_factor, nominal_ct * 0.05),
            nominal_ct * 0.7, nominal_ct * 1.8
        ))

        humidity = float(np.clip(self._rng.normal(50, 8), 20, 90))
        energy = float(np.clip(
            self._rng.normal(cycle_time * 0.025 * (1 + (temperature - base_temp) * 0.01), 0.5),
            0.5, 15.0
        ))

        return {
            "product_id": unit_id,
            "batch_id": batch_id,
            "station_id": sid,
            "temperature": round(temperature, 2),
            "pressure": round(pressure, 3),
            "machine_speed": round(speed, 1),
            "vibration": round(vibration, 3),
            "humidity": round(humidity, 1),
            "cycle_time_sec": round(cycle_time, 2),
            "tool_wear_percent": round(tool_wear, 1),
            "energy_consumption_kwh": round(energy, 3),
        }

    # ------------------------------------------------------------------
    # Defect determination (based on process conditions + patterns)
    # ------------------------------------------------------------------

    def _determine_defect(self, variant, station, batch_idx, proc) -> Tuple[dict, bool]:
        sid = station["id"]

        # Novel defect?
        if self._rng.random() < self.cfg["novel_defect_rate"]:
            return {"type": "Novel_Defect", "family": "Unknown", "severity": "critical",
                    "resolution": "scrap", "cost": 300}, True

        # Base defect rate from variant
        base_rate = variant["base_defect_rate"]

        # Increase for drifted conditions
        multiplier = 1.0

        # Pattern 1: S3 temp drift → Crack + Discoloration
        if sid == "S3" and batch_idx >= PATTERNS["s3_temp_drift"]["start_batch_idx"]:
            drift_batches = batch_idx - PATTERNS["s3_temp_drift"]["start_batch_idx"]
            multiplier += drift_batches * 0.15

        # Pattern 2: S2 vibration → Scratch + Surface_Pit
        if sid == "S2" and batch_idx >= PATTERNS["s2_vibration_drift"]["start_batch_idx"]:
            drift_batches = batch_idx - PATTERNS["s2_vibration_drift"]["start_batch_idx"]
            multiplier += drift_batches * 0.10

        # Pattern 4: Material batch issue
        if batch_idx in PATTERNS["material_batch_issue"]["batches"] and sid == "S1":
            multiplier += PATTERNS["material_batch_issue"]["defect_rate_multiplier"]

        # General: high temperature → more thermal defects
        temp_deviation = proc["temperature"] - (station["temp_min"] + station["temp_max"]) / 2
        if temp_deviation > (station["temp_max"] - station["temp_min"]) * 0.3:
            multiplier += 0.3

        # High vibration → surface defects
        if proc["vibration"] > 3.0:
            multiplier += 0.25

        defect_prob = min(base_rate * multiplier, 0.45)

        if self._rng.random() > defect_prob:
            none_entry = next(d for d in self._defect_types if d["type"] == "None")
            return none_entry, False

        # Select defect type weighted by station affinity
        candidates = [d for d in self._defect_types if d["type"] != "None"]

        # Weight by station affinity
        weights = []
        for d in candidates:
            if d.get("station") == sid:
                w = 3.0
            elif d.get("station") is None:
                w = 0.5
            else:
                w = 0.5
            # Extra weight for pattern-specific defects
            if sid == "S3" and batch_idx >= PATTERNS["s3_temp_drift"]["start_batch_idx"]:
                if d["type"] in ["Crack", "Discoloration"]:
                    w *= 2.5
            if sid == "S2" and batch_idx >= PATTERNS["s2_vibration_drift"]["start_batch_idx"]:
                if d["type"] in ["Scratch", "Surface_Pit"]:
                    w *= 2.5
            if batch_idx in PATTERNS["material_batch_issue"]["batches"] and sid == "S1":
                if d["type"] == "Porosity":
                    w *= 5.0
            weights.append(w)

        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        chosen = self._rng.choice(len(candidates), p=probs)
        return candidates[chosen], False

    # ------------------------------------------------------------------
    # Anomaly score computation
    # ------------------------------------------------------------------

    def _compute_anomaly_score(self, defect_info, proc, station, batch_idx) -> float:
        is_defective = defect_info["type"] not in ("None",)
        if is_defective:
            base = self._rng.uniform(0.60, 0.95)
        else:
            base = self._rng.uniform(0.05, 0.35)

        # Temperature deviation boosts score
        temp_mid = (station["temp_min"] + station["temp_max"]) / 2
        temp_dev = abs(proc["temperature"] - temp_mid) / max(station["temp_max"] - temp_mid, 1)
        base += temp_dev * 0.15

        # Vibration
        base += max(0, proc["vibration"] - 2.5) * 0.05

        return float(np.clip(base + self._rng.normal(0, 0.03), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Production aggregates (station × batch)
    # ------------------------------------------------------------------

    def _gen_production_aggregates(self, products_df, inspection_df, process_df) -> pd.DataFrame:
        merged = products_df.merge(
            inspection_df[["product_id", "inspection_result", "defect_family"]],
            on="product_id", how="left"
        )
        merged = merged.merge(
            process_df[["product_id", "cycle_time_sec", "utilization_pct"] if "utilization_pct" in process_df.columns
                        else ["product_id", "cycle_time_sec"]],
            on="product_id", how="left"
        )

        rows = []
        for (batch_id, station_id), grp in merged.groupby(["batch_id", "station_id"]):
            n_total = len(grp)
            n_defective = (grp["inspection_result"] == "defective").sum()
            n_good = n_total - n_defective

            # Scrap vs rework split based on defect catalog
            scrap_families = {"Structural_Defect", "Material_Defect", "Unknown"}
            n_scrap = (grp["defect_family"].isin(scrap_families) &
                       (grp["inspection_result"] == "defective")).sum()
            n_rework = n_defective - n_scrap

            station_cfg = next((s for s in self._stations if s["id"] == station_id), None)
            if station_cfg is None:
                continue

            nominal_ct = station_cfg["cycle_sec"]
            avg_ct = float(process_df[process_df["station_id"] == station_id]["cycle_time_sec"].mean()
                           if "cycle_time_sec" in process_df.columns else nominal_ct)
            capacity_ph = 3600 / avg_ct if avg_ct > 0 else station_cfg["capacity_ph"]
            actual_tput = min(capacity_ph * 0.85, n_total / max(1, len(grp)) * capacity_ph)
            utilization = min(100.0, (actual_tput / station_cfg["capacity_ph"]) * 100)

            # Downtime: stations with higher defect rates have more downtime
            defect_rate = n_defective / max(n_total, 1)
            downtime_min = float(self._rng.normal(
                10 + defect_rate * 30, 5
            ))

            rows.append({
                "batch_id": batch_id,
                "station_id": station_id,
                "planned_units": n_total,
                "produced_units": n_total,
                "good_units": int(n_good),
                "defective_units": int(n_defective),
                "rework_units": int(n_rework),
                "scrap_units": int(n_scrap),
                "downtime_minutes": round(max(0, downtime_min), 1),
                "utilization_percent": round(utilization, 1),
                "capacity_units_per_hour": round(station_cfg["capacity_ph"], 1),
                "actual_throughput_units_per_hour": round(actual_tput, 1),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Downtime events
    # ------------------------------------------------------------------

    def _gen_downtime_events(self, production_df) -> pd.DataFrame:
        reasons = ["Machine_Failure", "Maintenance", "Material_Shortage",
                   "Changeover", "Quality_Hold", "Operator_Delay"]
        severities = ["low", "medium", "high"]
        rows = []
        event_id = 1
        start_ts = self._start_date

        for (batch_id, station_id), row in production_df.groupby(["batch_id", "station_id"]).first().iterrows():
            # Number of downtime events proportional to total downtime
            total_dt = float(row["downtime_minutes"]) if "downtime_minutes" in row else 0
            if total_dt <= 0:
                continue
            n_events = max(1, int(self._rng.poisson(total_dt / 20)))

            for _ in range(n_events):
                duration = max(1, self._rng.exponential(total_dt / n_events))
                event_start = start_ts + timedelta(
                    hours=self._rng.uniform(0, 8)
                )
                reason = self._rng.choice(reasons)
                severity = self._rng.choice(severities, p=[0.5, 0.35, 0.15])

                rows.append({
                    "downtime_id": f"DT{event_id:05d}",
                    "batch_id": batch_id,
                    "station_id": station_id,
                    "start_time": event_start.isoformat(),
                    "duration_minutes": round(duration, 1),
                    "downtime_reason": reason,
                    "severity": severity,
                })
                event_id += 1

            start_ts += timedelta(hours=self._rng.uniform(0.5, 3))

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Economics
    # ------------------------------------------------------------------

    def _gen_economics(self, products_df, inspection_df) -> pd.DataFrame:
        econ_cfg = self.cfg["economics"]
        merged = products_df.merge(
            inspection_df[["product_id", "inspection_result", "defect_family", "defect_type"]],
            on="product_id", how="left"
        )
        rows = []
        for _, row in merged.iterrows():
            variant = next((v for v in self._variants if v["id"] == row["product_variant"]), self._variants[0])
            is_defective = row["inspection_result"] == "defective"

            # Defect cost lookup
            defect_entry = next(
                (d for d in self._defect_types if d["type"] == row.get("defect_type", "None")),
                None
            )
            scrap_cost = 0.0
            rework_cost = 0.0
            if is_defective and defect_entry:
                if defect_entry.get("resolution") == "scrap":
                    scrap_cost = float(defect_entry.get("cost", 200))
                elif defect_entry.get("resolution") == "rework":
                    rework_cost = float(defect_entry.get("cost", 60))

            material_cost = float(self._rng.normal(variant["material"], variant["material"] * 0.05))
            processing_cost = float(self._rng.normal(econ_cfg["processing_cost_base"], 5))
            energy_cost = float(self._rng.normal(econ_cfg["energy_cost_base"], 2))
            labor_cost = float(self._rng.normal(econ_cfg["labor_cost_base"], 4))
            inspection_cost = econ_cfg["inspection_cost"]
            selling_price = float(self._rng.normal(variant["price"], variant["price"] * 0.02))

            total_cost = (material_cost + processing_cost + energy_cost +
                          labor_cost + rework_cost + scrap_cost + inspection_cost)
            profit = selling_price - total_cost
            margin_pct = (profit / selling_price * 100) if selling_price > 0 else 0

            rows.append({
                "product_id": row["product_id"],
                "batch_id": row["batch_id"],
                "product_variant": row["product_variant"],
                "material_cost": round(material_cost, 2),
                "processing_cost": round(processing_cost, 2),
                "energy_cost": round(energy_cost, 2),
                "labor_cost": round(labor_cost, 2),
                "rework_cost": round(rework_cost, 2),
                "scrap_cost": round(scrap_cost, 2),
                "total_cost": round(total_cost, 2),
                "selling_price": round(selling_price, 2),
                "profit": round(profit, 2),
                "profit_margin_percent": round(margin_pct, 2),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Batch summaries (computed from actual unit-level data)
    # ------------------------------------------------------------------

    def _summarize_batches(self, products_df, inspection_df, process_df) -> pd.DataFrame:
        merged = products_df.merge(
            inspection_df[["product_id", "inspection_result"]],
            on="product_id", how="left"
        ).merge(
            process_df[["product_id", "cycle_time_sec", "temperature",
                         "pressure", "machine_speed"]],
            on="product_id", how="left"
        )

        rows = []
        batch_start_ts = self._start_date
        for i, batch_id in enumerate(self._batch_ids):
            grp = merged[merged["batch_id"] == batch_id]
            if len(grp) == 0:
                continue

            variant_id = grp["product_variant"].iloc[0]
            n_total = len(grp)
            n_defective = (grp["inspection_result"] == "defective").sum()
            defect_rate = n_defective / n_total * 100

            duration_h = max(1, n_total / 40)
            batch_end_ts = batch_start_ts + timedelta(hours=duration_h)

            # Drift score: mean absolute deviation from nominal
            temp_col = grp["temperature"]
            station_cfg = self._stations[i % len(self._stations)]
            temp_nominal = (station_cfg["temp_min"] + station_cfg["temp_max"]) / 2
            temp_range = max(1, station_cfg["temp_max"] - station_cfg["temp_min"])
            drift_score = float((temp_col - temp_nominal).abs().mean() / temp_range)

            rows.append({
                "batch_id": batch_id,
                "product_variant": variant_id,
                "material_batch": f"MAT-{batch_id}",
                "start_time": batch_start_ts.isoformat(),
                "end_time": batch_end_ts.isoformat(),
                "planned_quantity": n_total,
                "actual_quantity": n_total,
                "defect_rate_percent": round(defect_rate, 2),
                "rework_rate_percent": round(defect_rate * 0.6, 2),
                "scrap_rate_percent": round(defect_rate * 0.4, 2),
                "average_cycle_time_sec": round(float(grp["cycle_time_sec"].mean()), 2),
                "average_temperature": round(float(grp["temperature"].mean()), 2),
                "average_pressure": round(float(grp["pressure"].mean()), 3),
                "average_machine_speed": round(float(grp["machine_speed"].mean()), 1),
                "process_drift_score": round(drift_score, 4),
            })

            batch_start_ts = batch_end_ts + timedelta(hours=self._rng.uniform(0.5, 2))

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_location(self) -> str:
        locations = ["upper-left", "upper-right", "lower-left", "lower-right",
                     "center", "edge", "surface", "corner"]
        return str(self._rng.choice(locations))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InspectIQ Synthetic Dataset Generator")
    parser.add_argument("--output-dir", default="data/synthetic",
                        help="Output directory for CSV files")
    parser.add_argument("--n-units", type=int, default=DEFAULT_CONFIG["n_units"],
                        help="Total number of production units")
    parser.add_argument("--n-batches", type=int, default=DEFAULT_CONFIG["n_batches"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["n_units"] = args.n_units
    config["n_batches"] = args.n_batches
    config["seed"] = args.seed

    gen = SyntheticDataGenerator(config)
    gen.generate_all(args.output_dir)
    logger.info("✓ Dataset generated in %s", args.output_dir)


if __name__ == "__main__":
    main()
