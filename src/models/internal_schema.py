"""
src/models/internal_schema.py

Internal data model for InspectIQ.
These Pydantic models define the LOGICAL schema used by all engines.
They are NOT tied to any external dataset's column names.
Column mapping is handled by src/ingestion/schema_adapter.py.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class DecisionState(str, Enum):
    """Possible inspection decision states."""
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REVIEW = "REVIEW"
    NOVEL  = "NOVEL"
    PENDING = "PENDING"   # not yet processed


class DowntimeReason(str, Enum):
    MACHINE_FAILURE  = "Machine_Failure"
    MAINTENANCE      = "Maintenance"
    MATERIAL_SHORTAGE = "Material_Shortage"
    CHANGEOVER       = "Changeover"
    QUALITY_HOLD     = "Quality_Hold"
    OPERATOR_DELAY   = "Operator_Delay"
    UNKNOWN          = "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Core internal records
# ─────────────────────────────────────────────────────────────────────────────

class Unit(BaseModel):
    """
    One inspected production unit.
    All field names here are INTERNAL — they are never read directly from
    the dataset. The schema_adapter maps external columns to these fields.
    """
    unit_id: str
    variant_id: Optional[str] = None
    batch_id: Optional[str] = None
    station_id: Optional[str] = None
    shift: Optional[str] = None
    operator_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    # Inspection results (may be pre-computed from dataset or from Vision Engine)
    inspection_result: Optional[str] = None      # "good" / "defective" / "rework"
    defect_family: Optional[str] = None          # e.g. "Scratch", "Crack", "None"
    defect_severity: Optional[str] = None        # "minor", "major", "critical"
    defect_location: Optional[str] = None        # free text

    # Vision Engine outputs (populated after processing)
    anomaly_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    decision_state: DecisionState = DecisionState.PENDING
    is_novel: bool = False
    prediction_set: List[str] = Field(default_factory=list)

    # Localization outputs
    heatmap_path: Optional[str] = None
    mask_path: Optional[str] = None
    bounding_boxes: List[Dict[str, int]] = Field(default_factory=list)

    # Image reference
    image_path: Optional[str] = None

    # Metadata
    extra_fields: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class ProductionEvent(BaseModel):
    """
    A production event tied to a unit and station.
    One row per unit-station interaction.
    """
    unit_id: str
    station_id: str
    batch_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    cycle_time_sec: Optional[float] = Field(None, ge=0)
    state: Optional[str] = None       # "active", "blocked", "starved", "idle"
    downtime_minutes: Optional[float] = Field(None, ge=0)
    is_changeover: bool = False
    is_rework: bool = False


class ProcessObservation(BaseModel):
    """
    A process parameter measurement at a station, linked to a unit/batch.
    The parameter_name field is dynamic — populated from schema mapping.
    """
    station_id: str
    parameter_name: str
    parameter_value: float
    unit_id: Optional[str] = None
    batch_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class EconomicRecord(BaseModel):
    """
    Economic data for a production unit.
    Missing fields default to None — fallback values are applied by the
    Economics Engine (from economics.yaml) and labelled as ASSUMPTIONS.
    """
    unit_id: Optional[str] = None
    batch_id: Optional[str] = None
    variant_id: Optional[str] = None

    selling_price: Optional[float] = None
    material_cost: Optional[float] = None
    processing_cost: Optional[float] = None
    energy_cost: Optional[float] = None
    labor_cost: Optional[float] = None
    rework_cost: Optional[float] = None
    scrap_cost: Optional[float] = None
    inspection_cost: Optional[float] = None
    downtime_cost: Optional[float] = None
    total_cost: Optional[float] = None
    profit: Optional[float] = None
    profit_margin_pct: Optional[float] = None

    # Track which fields were estimated from fallback (shown in UI as ASSUMPTION)
    assumed_fields: List[str] = Field(default_factory=list)


class StationConfig(BaseModel):
    """Reference information about a production station."""
    station_id: str
    station_name: Optional[str] = None
    process_type: Optional[str] = None
    nominal_cycle_time_sec: Optional[float] = None
    max_capacity_per_hour: Optional[float] = None
    normal_temp_min: Optional[float] = None
    normal_temp_max: Optional[float] = None
    normal_pressure_min: Optional[float] = None
    normal_pressure_max: Optional[float] = None


class BatchSummary(BaseModel):
    """Aggregate summary for a production batch."""
    batch_id: str
    variant_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    planned_quantity: Optional[int] = None
    actual_quantity: Optional[int] = None
    good_units: Optional[int] = None
    defect_rate_pct: Optional[float] = None
    rework_rate_pct: Optional[float] = None
    scrap_rate_pct: Optional[float] = None
    avg_cycle_time_sec: Optional[float] = None
    process_drift_score: Optional[float] = None


class ProductionAggregate(BaseModel):
    """
    Station × Batch level production aggregate.
    One row per station-batch combination.
    """
    batch_id: str
    station_id: str
    planned_units: Optional[int] = None
    produced_units: Optional[int] = None
    good_units: Optional[int] = None
    defective_units: Optional[int] = None
    rework_units: Optional[int] = None
    scrap_units: Optional[int] = None
    downtime_minutes: Optional[float] = None
    utilization_pct: Optional[float] = None
    capacity_per_hour: Optional[float] = None
    throughput_per_hour: Optional[float] = None


class DowntimeEvent(BaseModel):
    """A downtime event at a station."""
    downtime_id: Optional[str] = None
    station_id: str
    batch_id: Optional[str] = None
    start_time: Optional[datetime] = None
    duration_minutes: float
    reason: str = DowntimeReason.UNKNOWN
    severity: Optional[str] = None


class DefectCatalogEntry(BaseModel):
    """Reference entry in the defect catalog."""
    defect_type: str
    defect_family: str
    typical_station: Optional[str] = None
    possible_causes: Optional[str] = None
    severity: Optional[str] = None
    resolution: Optional[str] = None    # "scrap" or "rework"
    typical_cost: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Engine output records
# ─────────────────────────────────────────────────────────────────────────────

class RootCauseHypothesis(BaseModel):
    """A ranked root-cause hypothesis generated by the Root-Cause Engine."""
    rank: int
    defect_family: str
    hypothesis: str
    evidence: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    effect_size: Optional[float] = None
    adjusted_p_value: Optional[float] = None
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None
    statistical_support: Dict[str, Any] = Field(default_factory=dict)
    alternative_explanations: List[str] = Field(default_factory=list)
    confounders: List[str] = Field(default_factory=list)


class BottleneckResult(BaseModel):
    """Bottleneck detection result for a time window."""
    station_id: str
    bottleneck_score: float
    utilization_pct: Optional[float] = None
    blocked_time_pct: Optional[float] = None
    starved_time_pct: Optional[float] = None
    avg_cycle_time_sec: Optional[float] = None
    wip_upstream: Optional[float] = None
    throughput_loss_units: Optional[float] = None
    value_of_relief: Optional[float] = None   # estimated margin gain if bottleneck relieved


class SimulationResult(BaseModel):
    """Output from a SimPy flow simulation run."""
    scenario_label: str
    is_baseline: bool
    throughput_per_hour: float
    wip_avg: float
    utilization_by_station: Dict[str, float] = Field(default_factory=dict)
    scrap_units: float
    rework_units: float
    downtime_minutes: float
    total_margin: Optional[float] = None
    margin_per_unit: Optional[float] = None
    # Deltas vs baseline (populated by ScenarioRunner)
    delta_throughput: Optional[float] = None
    delta_margin: Optional[float] = None
    delta_scrap: Optional[float] = None
    advisory_label: str = "SIMULATED / ADVISORY"


class Recommendation(BaseModel):
    """
    An evidence-based, advisory recommendation.
    Language must be advisory — never imperative.
    """
    rank: int
    action: str                    # "Simulation indicates that..." phrasing
    evidence: List[str]
    expected_impact: Dict[str, Any]   # {metric: delta}
    confidence: float = Field(ge=0.0, le=1.0)
    assumptions: List[str]
    alternative_explanations: List[str]
    status: str = "SIMULATED / ADVISORY"
    linked_hypothesis: Optional[RootCauseHypothesis] = None
    linked_bottleneck: Optional[BottleneckResult] = None


class DataQualityReport(BaseModel):
    """Result of data validation step."""
    total_records: int
    valid_records: int
    invalid_records: int
    missing_required_fields: Dict[str, int] = Field(default_factory=dict)
    null_pct_by_field: Dict[str, float] = Field(default_factory=dict)
    duplicate_records: int = 0
    type_errors: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    is_valid: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Session state (shared across engines in Streamlit app)
# ─────────────────────────────────────────────────────────────────────────────

class InspectIQSession(BaseModel):
    """
    Holds all loaded and computed data for the current session.
    Stored in Streamlit session_state.
    """
    # Raw DataFrames (as dicts for Pydantic serialisation; in practice held as pd.DataFrame)
    units_df: Optional[Any] = None
    production_df: Optional[Any] = None
    process_df: Optional[Any] = None
    economics_df: Optional[Any] = None
    stations_df: Optional[Any] = None
    batches_df: Optional[Any] = None
    downtime_df: Optional[Any] = None
    defect_catalog_df: Optional[Any] = None

    # Derived
    data_quality_report: Optional[DataQualityReport] = None
    vision_results: Optional[Any] = None       # pd.DataFrame
    hypotheses: List[RootCauseHypothesis] = Field(default_factory=list)
    bottleneck_results: List[BottleneckResult] = Field(default_factory=list)
    baseline_simulation: Optional[SimulationResult] = None
    scenario_simulation: Optional[SimulationResult] = None
    recommendations: List[Recommendation] = Field(default_factory=list)

    # Applied filters (populated from sidebar)
    filter_variants: List[str] = Field(default_factory=list)
    filter_batches: List[str] = Field(default_factory=list)
    filter_stations: List[str] = Field(default_factory=list)
    filter_defect_families: List[str] = Field(default_factory=list)
    filter_shifts: List[str] = Field(default_factory=list)

    # Processing flags
    data_loaded: bool = False
    vision_run: bool = False
    rootcause_run: bool = False
    flow_run: bool = False
    economics_run: bool = False

    class Config:
        arbitrary_types_allowed = True
