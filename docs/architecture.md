# InspectIQ — Architecture & Design Notes

## System Overview

InspectIQ is a **software-only, advisory** decision-support system that connects
visual defect inspection → root-cause analysis → flow/bottleneck analysis → economics → what-if simulation.

## Core Design Principle: No Hardcoded Business Logic

The application **never** embeds:
- Station IDs, batch IDs, variant names, defect category names
- Threshold values, cost values, selling prices
- Dashboard KPI values, recommendation text
- Production topology, number of stations, number of variants

All such values originate from:
1. Uploaded datasets (primary source)
2. Schema-mapped through `configs/schema_mapping.yaml`
3. Config-driven fallback assumptions (explicitly labelled in UI)
4. Model outputs and simulation results

## Module Architecture

```
Dataset Upload
    ↓
configs/schema_mapping.yaml      ← maps external column names
    ↓
src/ingestion/schema_adapter.py  ← renames columns
    ↓
src/ingestion/validator.py       ← validates quality
    ↓
src/ingestion/data_loader.py     ← loads all tables
    ↓
        ┌──────────────────┬──────────────────┐
        ↓                  ↓                  ↓
src/vision/           src/rootcause/      src/flow/
vision_engine.py      rootcause_engine.py flow_engine.py
        ↓                  ↓                  ↓
src/uncertainty/      Hypotheses          Bottleneck +
decision_engine.py                        Simulation
        ↓                  ↓                  ↓
        └──────────────────┴──────────────────┘
                           ↓
              src/economics/economics_engine.py
                           ↓
            src/recommend/recommendation_engine.py
                           ↓
              src/engine_runner.py  (orchestrator)
                           ↓
              dashboard/app.py  (Streamlit UI)
```

## Decision States

Every inspected unit gets one of four states:
- `ACCEPT` — anomaly score ≤ low_threshold, no defect
- `REJECT` — anomaly score ≥ high_threshold, known defect
- `REVIEW` — score in ambiguous zone, or multiple possible defect classes
- `NOVEL` — anomaly detected but outside known defect families

Thresholds are calibrated from data by the Economics Engine (minimizing expected cost).

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/schema_mapping.yaml` | Maps external columns → internal schema |
| `configs/vision.yaml` | Vision engine model and feature parameters |
| `configs/uncertainty.yaml` | Conformal alpha, novelty threshold, decision states |
| `configs/rootcause.yaml` | Statistical test parameters, SHAP settings |
| `configs/flow.yaml` | Bottleneck detection, SimPy simulation settings |
| `configs/economics.yaml` | Fallback cost assumptions (labelled as ASSUMPTIONS) |
| `configs/simulation.yaml` | What-if scenario parameter ranges |
| `configs/dashboard.yaml` | UI layout, colors, chart settings |

## Data Flow (Single Unit)

1. Unit `U-XXXXX` processed by ingestion layer
2. Schema adapter translates external column names → internal fields
3. Vision Engine scores unit: `anomaly_score ∈ [0,1]` + heatmap
4. Decision Engine assigns `ACCEPT/REJECT/REVIEW/NOVEL` based on calibrated thresholds
5. Root-Cause Engine links defect pattern to process parameters (statistical hypothesis)
6. Flow Engine determines if unit's station is a bottleneck
7. Economics Engine estimates cost impact
8. Recommendation Engine generates advisory action (SIMULATED / ADVISORY)

## Evaluation Metrics

- **Vision**: AUROC, AUPRC, F1, False Reject Rate, localization quality
- **Uncertainty**: Coverage calibration, abstention rate, novelty precision
- **Root Cause**: Hypothesis precision/recall on synthetic injected faults
- **Flow**: Bottleneck detection accuracy, simulation calibration error
- **Economics**: Margin forecast accuracy, threshold optimization gap

## Scope Limitations (by design)

- **Software only** — no camera, PLC, or hardware connections
- **Offline/replay** — processes pre-recorded datasets
- **Advisory** — all recommendations labelled SIMULATED / ADVISORY
- **Human-in-the-loop** — REVIEW and NOVEL cases require human review
