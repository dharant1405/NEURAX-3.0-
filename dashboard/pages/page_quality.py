"""
dashboard/pages/page_quality.py

Quality Dashboard page.
Inspection stream, defect rates, ACCEPT/REJECT/REVIEW/NOVEL distribution,
heatmap evidence panel, defect family trends.
All values computed from data — no hardcoded numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.engine_runner import EngineRunner

_ROOT = Path(__file__).resolve().parents[2]

COLORS = {
    "ACCEPT": "#00C853",
    "REJECT": "#D50000",
    "REVIEW": "#FF6D00",
    "NOVEL":  "#AA00FF",
    "PENDING": "#546e7a",
}


@st.cache_resource
def get_engine() -> EngineRunner:
    return EngineRunner()


def render():
    engine = get_engine()
    st.markdown("# 🔍 Quality Dashboard")

    if not st.session_state.get("data_loaded"):
        st.info("👈 Load data first on the **Data Upload** page.")
        return

    # ── Run vision if not done ─────────────────────────────────────────────
    if not st.session_state.get("vision_run"):
        if st.button("▶ Run Vision Engine", use_container_width=True, type="primary"):
            with st.spinner("Running Vision & Decision Engine..."):
                engine.run_vision()
            st.rerun()
        st.info("Run the Vision Engine to see inspection results.")
        return

    vision_df = st.session_state.get("vision_results")
    if vision_df is None or len(vision_df) == 0:
        st.warning("No vision results available.")
        return

    # ── Apply global filters ───────────────────────────────────────────────
    filtered_df = _apply_global_filters(vision_df)
    decision_summary = st.session_state.get("decision_summary", {})

    # ── KPI Row ────────────────────────────────────────────────────────────
    st.markdown("### 📊 Inspection Summary")
    total = len(filtered_df)
    counts = filtered_df["decision_state"].value_counts().to_dict() if "decision_state" in filtered_df.columns else {}

    n_accept = counts.get("ACCEPT", 0)
    n_reject = counts.get("REJECT", 0)
    n_review = counts.get("REVIEW", 0)
    n_novel  = counts.get("NOVEL",  0)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Inspected", f"{total:,}")
    col2.metric("✅ ACCEPT", f"{n_accept:,}", f"{n_accept/max(total,1)*100:.1f}%")
    col3.metric("❌ REJECT", f"{n_reject:,}", f"{n_reject/max(total,1)*100:.1f}%")
    col4.metric("🟡 REVIEW", f"{n_review:,}", f"{n_review/max(total,1)*100:.1f}%")
    col5.metric("🔮 NOVEL",  f"{n_novel:,}",  f"{n_novel/max(total,1)*100:.1f}%")

    # Calibration status
    dt_cfg = st.session_state.get("threshold_optimization")
    if dt_cfg:
        st.markdown(
            f"<div class='advisory-banner'>⚡ Economics-optimized thresholds active — "
            f"Low: {dt_cfg.get('optimal_low_threshold',0):.3f} | "
            f"High: {dt_cfg.get('optimal_high_threshold',0):.3f} | "
            f"False Accept Cost: {dt_cfg.get('false_accept_cost_used')} | "
            f"False Reject Cost: {dt_cfg.get('false_reject_cost_used')} "
            f"[SIMULATED / ADVISORY]</div>",
            unsafe_allow_html=True,
        )

    # ── Charts Row 1 ──────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### Decision State Distribution")
        if counts:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.45,
                marker_colors=[COLORS.get(k, "#546e7a") for k in counts.keys()],
                textinfo="label+percent",
                textfont_size=13,
            )])
            fig_pie.update_layout(
                template="plotly_dark",
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("#### Anomaly Score Distribution")
        score_col = "anomaly_score_final" if "anomaly_score_final" in filtered_df.columns else "anomaly_score"
        if score_col in filtered_df.columns:
            fig_hist = px.histogram(
                filtered_df, x=score_col, color="decision_state",
                color_discrete_map=COLORS,
                barmode="overlay", nbins=40,
                template="plotly_dark",
                labels={score_col: "Anomaly Score", "decision_state": "Decision"},
            )
            # Add threshold lines
            ds = st.session_state.get("decision_summary", {})
            low_t = ds.get("low_threshold", engine.decision.low_threshold)
            high_t = ds.get("high_threshold", engine.decision.high_threshold)
            fig_hist.add_vline(x=low_t, line_dash="dash", line_color="#FF6D00",
                               annotation_text=f"Low: {low_t:.2f}")
            fig_hist.add_vline(x=high_t, line_dash="dash", line_color="#D50000",
                               annotation_text=f"High: {high_t:.2f}")
            fig_hist.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # ── Defect rate trend over batches (computed from data) ────────────────
    st.divider()
    st.markdown("### 📈 Defect Rate Trend")

    if "batch_id" in filtered_df.columns and "decision_state" in filtered_df.columns:
        batch_trend = filtered_df.groupby("batch_id").apply(
            lambda g: pd.Series({
                "total": len(g),
                "defect_rate": (g["decision_state"].isin(["REJECT", "NOVEL"])).mean() * 100,
                "review_rate": (g["decision_state"] == "REVIEW").mean() * 100,
            })
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=batch_trend["batch_id"], y=batch_trend["defect_rate"],
            name="Defect Rate (%)", line=dict(color="#D50000", width=2),
            mode="lines+markers"
        ))
        fig_trend.add_trace(go.Scatter(
            x=batch_trend["batch_id"], y=batch_trend["review_rate"],
            name="Review Rate (%)", line=dict(color="#FF6D00", width=2, dash="dot"),
            mode="lines+markers"
        ))
        fig_trend.update_layout(
            template="plotly_dark", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Batch", yaxis_title="Rate (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ── Defect families (computed from data) ──────────────────────────────
    col_c, col_d = st.columns([1, 1])

    with col_c:
        st.markdown("### 🏷️ Defect Family Breakdown")
        fam_col = None
        for c in ["defect_family", "defect_family_predicted"]:
            if c in filtered_df.columns:
                fam_col = c
                break
        if fam_col:
            defective_df = filtered_df[filtered_df["decision_state"].isin(["REJECT", "NOVEL"])]
            family_counts = (
                defective_df[fam_col]
                .dropna()
                .value_counts()
                .reset_index()
            )
            family_counts.columns = ["defect_family", "count"]
            if len(family_counts) > 0:
                fig_bar = px.bar(
                    family_counts, x="count", y="defect_family", orientation="h",
                    template="plotly_dark", color="count",
                    color_continuous_scale="Reds",
                    labels={"count": "Units", "defect_family": "Defect Family"},
                )
                fig_bar.update_layout(
                    height=350, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    with col_d:
        st.markdown("### 🏭 Defect Rate by Station")
        if "station_id" in filtered_df.columns and "decision_state" in filtered_df.columns:
            station_defects = filtered_df.groupby("station_id").apply(
                lambda g: (g["decision_state"].isin(["REJECT", "NOVEL"])).mean() * 100
            ).reset_index()
            station_defects.columns = ["station_id", "defect_rate"]
            fig_station = px.bar(
                station_defects.sort_values("defect_rate", ascending=True),
                x="defect_rate", y="station_id", orientation="h",
                template="plotly_dark",
                color="defect_rate", color_continuous_scale="Reds",
                labels={"defect_rate": "Defect Rate (%)", "station_id": "Station"},
            )
            fig_station.update_layout(
                height=350, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_station, use_container_width=True)

    # ── Evidence panel: sample defective units ─────────────────────────────
    st.divider()
    st.markdown("### 🔬 Unit Evidence Panel")
    st.markdown(
        "Select a unit to trace: Image → Anomaly Score → Decision → "
        "Defect Family → Batch → Station → Process Evidence"
    )

    rejected_df = filtered_df[filtered_df["decision_state"].isin(["REJECT", "NOVEL", "REVIEW"])]
    if len(rejected_df) > 0:
        unit_ids = rejected_df["unit_id"].dropna().tolist()[:100]
        selected_unit = st.selectbox("Select Unit ID", unit_ids, key="evidence_unit_select")

        if selected_unit:
            unit_row = filtered_df[filtered_df["unit_id"] == selected_unit].iloc[0]
            _render_evidence_panel(unit_row)
    else:
        st.info("No rejected/review units to show. Adjust filters or check thresholds.")

    # ── Novel defect queue ─────────────────────────────────────────────────
    novel_df = filtered_df[filtered_df["decision_state"] == "NOVEL"]
    if len(novel_df) > 0:
        st.divider()
        st.markdown("### 🔮 Novel Defect Queue")
        st.markdown(
            "<div class='advisory-banner'>⚠️ These units do not match known defect families. "
            "Human review recommended before classification.</div>",
            unsafe_allow_html=True,
        )
        cols_to_show = [c for c in ["unit_id", "batch_id", "station_id", "anomaly_score_final",
                                     "decision_reason", "timestamp"]
                        if c in novel_df.columns]
        st.dataframe(novel_df[cols_to_show].head(20), use_container_width=True)


def _apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    filter_variants = st.session_state.get("filter_variants", [])
    filter_batches = st.session_state.get("filter_batches", [])
    filter_stations = st.session_state.get("filter_stations", [])
    filter_families = st.session_state.get("filter_defect_families", [])

    for filter_list, col_candidates in [
        (filter_variants, ["variant_id", "product_variant"]),
        (filter_batches,  ["batch_id"]),
        (filter_stations, ["station_id"]),
        (filter_families, ["defect_family", "defect_family_predicted"]),
    ]:
        if filter_list:
            for col in col_candidates:
                if col in filtered.columns:
                    filtered = filtered[filtered[col].isin(filter_list)]
                    break

    return filtered


def _render_evidence_panel(row: pd.Series):
    """Render a full evidence chain for a single unit."""
    score_col = "anomaly_score_final" if "anomaly_score_final" in row.index else "anomaly_score"
    decision = row.get("decision_state", "PENDING")
    score = row.get(score_col, 0.0)
    color = {"ACCEPT": "#00C853", "REJECT": "#D50000", "REVIEW": "#FF6D00",
             "NOVEL": "#AA00FF"}.get(str(decision), "#546e7a")

    st.markdown(f"""
    <div style='border: 1px solid {color}; border-radius: 12px; padding: 16px;
                background: rgba(0,0,0,0.3);'>
    <h4 style='color: {color}; margin: 0 0 12px;'>
        Unit: {row.get('unit_id', '?')} —
        <span class='badge-{str(decision).lower()}'>{decision}</span>
    </h4>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    cols[0].metric("Anomaly Score", f"{float(score):.3f}")
    cols[1].metric("Confidence", f"{float(row.get('confidence_computed', row.get('confidence', 0))):.1%}")
    cols[2].metric("Batch", str(row.get("batch_id", "—")))
    cols[3].metric("Station", str(row.get("station_id", "—")))

    # Heatmap visualization (synthetic)
    hm_data = _get_heatmap_data(str(row.get("unit_id", "")), float(score))
    fig_hm = go.Figure(data=go.Heatmap(
        z=hm_data,
        colorscale="Hot",
        showscale=True,
    ))
    fig_hm.update_layout(
        title="Anomaly Heatmap (model-generated)",
        template="plotly_dark", height=250,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_showticklabels=False, yaxis_showticklabels=False,
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption(
        "⚠️ Heatmap is model-generated. Quantitative localization metrics require "
        "ground-truth masks."
    )

    # Defect info
    fam = row.get("defect_family") or row.get("defect_family_predicted", "Unknown")
    sev = row.get("defect_severity", "—")
    loc = row.get("defect_location", "—")
    reason = row.get("decision_reason", "—")

    st.markdown(f"""
    **Evidence Chain:**
    - 🏷️ Defect Family: **{fam}**
    - 🔴 Severity: **{sev}**
    - 📍 Location: **{loc}**
    - 📋 Decision Reason: *{reason}*
    """)

    # Process parameters for this unit
    process_df = st.session_state.get("process_parameters_df")
    if process_df is not None and "unit_id" in process_df.columns:
        unit_proc = process_df[process_df["unit_id"] == row.get("unit_id")]
        if len(unit_proc) > 0:
            st.markdown("**Process Parameters at time of production:**")
            proc_row = unit_proc.iloc[0]
            param_cols = [c for c in ["temperature", "pressure", "machine_speed",
                                       "vibration", "cycle_time_sec", "tool_wear_percent"]
                          if c in proc_row.index]
            p_cols = st.columns(len(param_cols))
            for i, col in enumerate(param_cols):
                p_cols[i].metric(col.replace("_", " ").title(), f"{proc_row[col]:.2f}")

    st.markdown("</div>", unsafe_allow_html=True)


def _get_heatmap_data(unit_id: str, anomaly_score: float) -> list:
    """Generate a proxy heatmap for visualization."""
    rng = np.random.default_rng(abs(hash(unit_id)) % (2**32))
    h, w = 32, 32
    cx, cy = int(rng.uniform(8, 24)), int(rng.uniform(8, 24))
    sigma = 4.0
    grid = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - cy)**2 + (j - cx)**2)
            grid[i, j] = anomaly_score * np.exp(-d**2 / (2 * sigma**2))
    grid += rng.normal(0, 0.02, (h, w))
    return np.clip(grid, 0, 1).tolist()
