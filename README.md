# InspectIQ — From Pixel to Profit
 
**A defect-to-profit decision-support system for multi-stage manufacturing**
 
> **NEURAX Hackathon 3.0 · Domain 2: AI in Industry and Automation**
> **Problem Statement:** Visual Inspection & Defect Root-Cause Assistant
> **Checkpoint 1: README** · Software-only · Simulated · Advisory
 
---
 
## 1. Problem Understanding
 
On a high-throughput line with several stations and mixed product variants, four problems happen together but are usually solved in four separate tools:
 
| Layer | What goes wrong |
|---|---|
| **Quality** | Subtle defects are missed at line speed; good parts get rejected |
| **Process** | The same defect families keep returning with a batch, station or drifting setting |
| **Flow** | A station constrains the whole line, and the constraint shifts (downtime, changeovers, WIP, rework) |
| **Money** | Every reject, rework loop and stalled station changes margin |
 
**The ask:** not another image classifier or KPI dashboard, but **one system** that explains *what is wrong with a unit, where the defect is, what process condition likely caused it, where the line is constrained, what that costs, and what a change is worth*. It must also **flag uncertain or novel defects instead of guessing**, and stay **simulated and advisory** (no cameras, PLCs or machine control).
 
---
 
## 2. Our Idea: One Closed Loop
 
InspectIQ connects quality, flow and profit in a single loop. Inspection results feed the line simulation, the simulation feeds the profit model, and profit sets the accept/reject thresholds used by inspection.
 
```
Quality (defect & rework rates) → Flow (throughput, bottleneck) → Profit (margin)
        ↑                                                              │
        └──────────── cost-optimal accept / reject thresholds ←────────┘
```
 
---
 
## 3. System Architecture
 
![InspectIQ architecture](docs/architecture.svg)
 
| # | Module | What it does |
|---|---|---|
| 1 | **Ingestion & Schema Adapter** | Validates organizer data, maps columns to one internal schema, replays it as a simulated live stream |
| 2 | **Vision Engine** | Routes each image by product variant, scores it against a "normal" reference, produces a defect heatmap, mask and box |
| 3 | **Uncertainty Layer** | Calibrates confidence and outputs `ACCEPT`, `REJECT`, `REVIEW` or `NOVEL`; unknown defects go to a review queue |
| 4 | **Root-Cause Engine** | Links defect families to batch, station and process drift; returns ranked hypotheses with confidence and confounders |
| 5 | **Flow Engine** | Finds the shifting bottleneck and simulates throughput and losses; ranks stations by value of relief |
| 6 | **Economics Engine** | Computes unit economics and margin forecast; sets cost-optimal thresholds |
| 7 | **Recommendation Engine** | Runs simulated what-if scenarios and ranks advisory actions by margin impact |
| 8 | **Dashboard & API** | Four views: Quality, Root-Cause, Flow, Profit & What-If; each alert carries an evidence card |
 
---
 
## 4. Approach
 
**See — Vision.** Anomaly detection (PatchCore or EfficientAD via the Anomalib library) learns what a *good* unit looks like from normal images only, so it works when defect samples are rare and gives heatmaps for localization. A classifier is added only if labelled defects exist; otherwise anomalies are clustered into candidate defect families.
 
**Trust — Uncertainty.** Conformal prediction gives calibrated confidence. Ambiguous units go to `REVIEW`; patterns unlike any known defect become `NOVEL` and are queued for a human to name.
 
**Explain — Root cause.** Drift detection on process signals plus statistical tests (with false-discovery control) link defect families to batches, stations and parameters. Output is a *hypothesis* with evidence, never a claimed proof.
 
**Decide — Flow and profit.** The Active Period Method finds the momentary and shifting bottleneck from station-state data. A discrete-event simulation of the line estimates throughput and losses, and a margin model converts them into money. What-if scenarios turn this into ranked, advisory recommendations.
 
### Example: one unit's journey *(illustrative)*
 
1. Image of a unit arrives; the router picks the reference for its variant.
2. Vision finds an anomaly and draws a heatmap with a box around the hotspot.
3. Uncertainty confirms a confident `REJECT` for a known defect family.
4. Root-Cause links that family to a rise after a specific batch and a drift on an upstream station.
5. Flow shows the current bottleneck and how rework from this defect loads it.
6. Economics estimates the margin lost; Recommendation simulates a corrective setting and shows the gain.
---
 
## 5. What Makes It Different
 
- **Pixel-to-profit loop:** vision, root cause, flow and economics are one connected system, not separate tools.
- **Economics-aware decisions:** accept/reject thresholds minimize expected cost (missed defect vs. false reject vs. review), not a fixed cutoff.
- **Says "I don't know":** calibrated abstention with a novelty queue for unseen defect types.
- **Shifting bottlenecks:** finds where the constraint is *now* and what relieving it is worth.
- **Evidence-first:** every alert shows the image, heatmap, confidence, linked process evidence and simulated cost impact.
- **Cold-start friendly:** trains on normal images; labels are optional.
---
 
## 6. Tech Stack
 
| Area | Tools |
|---|---|
| Language & API | Python, FastAPI |
| Vision | PyTorch, Anomalib, OpenCV |
| Uncertainty & root cause | scikit-learn, SciPy, statsmodels, ruptures, LightGBM, SHAP |
| Flow & economics | SimPy, Active Period Method, pandas |
| Dashboard | Streamlit, Plotly |
| Reproducibility | Docker, pinned requirements, fixed seeds, pytest |
 
**Why it is feasible:** frozen pretrained backbones (no long training), fully offline on replayed data, loosely coupled modules that can be built in parallel.
 
---
 
## 7. Repository Structure
 
```text
inspectiq/
├── README.md
├── docs/architecture.svg
├── configs/            # schema mapping, thresholds, station and cost settings
├── data/               # raw (not committed) and synthetic test data
├── src/
│   ├── ingestion/  vision/  uncertainty/  rootcause/
│   └── flow/  economics/  recommend/  api/
├── dashboard/          # Streamlit app
├── tests/
└── requirements.txt · Dockerfile · Makefile
```
 
---
 
## 8. Roadmap
 
| Checkpoint | Focus |
|---|---|
| **1: README** | Problem understanding, architecture, approach (this document) |
| **2: Partial execution** | Ingestion, anomaly detection with heatmaps, first decisions, basic bottleneck view, early dashboard |
| **3: Full system** | Complete pipeline, calibrated uncertainty and novelty handling, root-cause and flow linkage, what-if recommendations, polished dashboard |
 
---
 
## 9. Scope and Responsible Use
 
- **Software-only:** no live camera feed, PLC connection, robotic sorting, machine control or line hardware.
- **Advisory:** every recommendation and profit estimate is labelled *simulated / advisory*.
- **Human in the loop:** uncertain and novel cases are routed to review.
- **Honest limits:** root-cause results are hypotheses, and flow and profit figures are simulation-based estimates.
 
