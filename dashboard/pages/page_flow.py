"""
dashboard/pages/page_flow.py

Flow Dashboard page.
Station utilization, bottleneck timeline, WIP, cycle times, loss waterfall.
All station IDs and values come from data — nothing hardcoded.
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


@st.cache_resource
def get_engine() -> EngineRunner:
    return EngineRunner()


def render():
    engine = get_engine()
    st.markdown("# ⚙️ Production Flow Analysis")

    if not st.session_state.get("data_loaded"):
        st.info("👈 Load data first on the **Data Upload** page.")
        return

    if not st.session_state.get("flow_run"):
        if st.button("▶ Run Flow Engine", use_container_width=True, type="primary"):
            with st.spinner("Analyzing production flow..."):
                engine.run_flow()
            st.rerun()
        st.info("Run the Flow Engine to see bottleneck and throughput analysis.")
        return

    production_df = st.session_state.get("production_df")
    bottleneck_results = st.session_state.get("bottleneck_results", [])
    flow_losses = st.session_state.get("flow_losses", {})
    baseline_sim = st.session_state.get("baseline_simulation", {})
    stations_df = st.session_state.get("stations_df")

    if production_df is None:
        st.warning("No production data loaded.")
        return

    # ── KPI Row ────────────────────────────────────────────────────────────
    st.markdown("### 📊 Production Summary")
    losses = flow_losses

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Units", f"{losses.get('total_produced_units', 0):,}")
    col2.metric("Good Units", f"{losses.get('total_good_units', 0):,}")
    col3.metric("Defect Rate", f"{losses.get('overall_defect_rate_pct', 0):.1f}%")
    col4.metric("Scrap Units", f"{losses.get('total_scrap_units', 0):,}")
    col5.metric("Downtime", f"{losses.get('total_downtime_hours', 0):.1f} h")

    # ── Bottleneck banner ──────────────────────────────────────────────────
    if bottleneck_results:
        primary_bn = bottleneck_results[0]
        sid = primary_bn.get("station_id", "?")
        sname = primary_bn.get("station_name", sid)
        bscore = primary_bn.get("bottleneck_score", 0)
        util = primary_bn.get("utilization_pct", 0)
        st.markdown(
            f"<div style='background: rgba(255,23,68,0.12); border: 1px solid #FF1744; "
            f"border-radius: 8px; padding: 12px 18px; color: #FF5252;'>"
            f"🔴 <b>Primary Bottleneck (detected from data): {sname} ({sid})</b> — "
            f"Bottleneck Score: {bscore:.3f} | Utilization: {util:.0f}%"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Station utilization chart ──────────────────────────────────────────
    st.divider()
    st.markdown("### 🏭 Station Utilization & Performance")

    if bottleneck_results:
        bn_df = pd.DataFrame(bottleneck_results)
        bottleneck_color = ["#FF1744" if r else "#00BCD4" for r in bn_df["is_primary_bottleneck"]]

        fig_util = go.Figure()
        fig_util.add_trace(go.Bar(
            x=bn_df["station_id"], y=bn_df["utilization_pct"],
            name="Utilization (%)",
            marker_color=bottleneck_color,
            text=bn_df["utilization_pct"].round(1).astype(str) + "%",
            textposition="auto",
        ))
        fig_util.add_trace(go.Bar(
            x=bn_df["station_id"], y=bn_df["throughput_per_hour"],
            name="Throughput (units/h)",
            marker_color="rgba(0,188,212,0.3)",
            yaxis="y2",
        ))
        fig_util.update_layout(
            template="plotly_dark", height=360, barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Station",
            yaxis=dict(title="Utilization (%)", side="left"),
            yaxis2=dict(title="Throughput (units/h)", side="right", overlaying="y"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_util, use_container_width=True)

    # ── Bottleneck rank table ─────────────────────────────────────────────
    st.markdown("### 🎯 Bottleneck Ranking")
    st.markdown(
        "<div class='advisory-banner'>🔄 Bottleneck scores computed from utilization, "
        "throughput gap, downtime, and cycle-time imbalance. Derived from data.</div>",
        unsafe_allow_html=True,
    )

    if bottleneck_results:
        cols_to_show = ["rank", "station_id", "station_name", "bottleneck_score",
                        "utilization_pct", "throughput_per_hour", "capacity_per_hour",
                        "downtime_minutes_total", "defect_rate_pct"]
        bn_show_df = pd.DataFrame(bottleneck_results)[
            [c for c in cols_to_show if c in pd.DataFrame(bottleneck_results).columns]
        ]
        st.dataframe(bn_show_df.round(2), use_container_width=True)

    # ── Cycle time comparison ──────────────────────────────────────────────
    st.divider()
    st.markdown("### ⏱️ Cycle Time by Station")

    if stations_df is not None and "nominal_cycle_time_sec" in stations_df.columns:
        ct_df = stations_df.copy()
        if "station_id" in ct_df.columns:
            # Actual from production
            if "capacity_units_per_hour" in production_df.columns:
                actual_ct = production_df.groupby("station_id").apply(
                    lambda g: 3600 / g["capacity_units_per_hour"].mean() if g["capacity_units_per_hour"].mean() > 0 else None
                ).reset_index()
                actual_ct.columns = ["station_id", "actual_cycle_time_sec"]
                ct_df = ct_df.merge(actual_ct, on="station_id", how="left")

            fig_ct = go.Figure()
            fig_ct.add_trace(go.Bar(
                x=ct_df["station_id"], y=ct_df["nominal_cycle_time_sec"],
                name="Nominal CT (s)", marker_color="rgba(0,188,212,0.5)",
            ))
            if "actual_cycle_time_sec" in ct_df.columns:
                fig_ct.add_trace(go.Bar(
                    x=ct_df["station_id"], y=ct_df["actual_cycle_time_sec"],
                    name="Actual CT (s)", marker_color="rgba(255,64,64,0.8)",
                ))
            fig_ct.update_layout(
                template="plotly_dark", height=320, barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Station", yaxis_title="Cycle Time (seconds)",
            )
            st.plotly_chart(fig_ct, use_container_width=True)
            st.caption(
                "Station with highest cycle time limits system throughput (Theory of Constraints). "
                "Red bars indicate actual measured cycle time vs nominal."
            )

    # ── Loss waterfall ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 💸 Production Loss Waterfall")
    st.markdown(
        "<div class='advisory-banner'>All cost estimates use data values where available, "
        "or labeled fallback assumptions. [ADVISORY]</div>",
        unsafe_allow_html=True,
    )

    if losses:
        total_planned = losses.get("total_planned_units", 1)
        waterfall_categories = []
        waterfall_values = []
        waterfall_measures = []

        waterfall_categories.append("Planned Output")
        waterfall_values.append(total_planned)
        waterfall_measures.append("absolute")

        for label, key in [
            ("Scrap Loss", "total_scrap_units"),
            ("Rework Capacity Loss", "total_rework_units"),
        ]:
            val = -losses.get(key, 0)
            if val != 0:
                waterfall_categories.append(label)
                waterfall_values.append(val)
                waterfall_measures.append("relative")

        waterfall_categories.append("Effective Output")
        waterfall_values.append(losses.get("total_good_units", 0))
        waterfall_measures.append("total")

        fig_wf = go.Figure(go.Waterfall(
            name="Production Loss",
            orientation="v",
            measure=waterfall_measures,
            x=waterfall_categories,
            y=waterfall_values,
            connector=dict(line=dict(color="#546e7a")),
            increasing=dict(marker_color="#00C853"),
            decreasing=dict(marker_color="#D50000"),
            totals=dict(marker_color="#00BCD4"),
        ))
        fig_wf.update_layout(
            template="plotly_dark", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Units",
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # ── Production by station trend ────────────────────────────────────────
    st.divider()
    st.markdown("### 📈 Defect Rate by Station & Batch")

    if "station_id" in production_df.columns and "batch_id" in production_df.columns:
        if "defective_units" in production_df.columns and "produced_units" in production_df.columns:
            trend_df = production_df.copy()
            trend_df["defect_rate_pct"] = (
                trend_df["defective_units"] / trend_df["produced_units"].replace(0, 1) * 100
            )
            fig_line = px.line(
                trend_df, x="batch_id", y="defect_rate_pct", color="station_id",
                template="plotly_dark",
                labels={"defect_rate_pct": "Defect Rate (%)", "batch_id": "Batch",
                        "station_id": "Station"},
                markers=True,
            )
            fig_line.update_layout(
                height=350, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_line, use_container_width=True)

    # ── Downtime breakdown ─────────────────────────────────────────────────
    downtime_df = st.session_state.get("downtime_df")
    if downtime_df is not None and len(downtime_df) > 0:
        st.divider()
        st.markdown("### ⏸️ Downtime Analysis")
        col_dt1, col_dt2 = st.columns(2)

        with col_dt1:
            if "downtime_reason" in downtime_df.columns:
                reason_counts = downtime_df.groupby("downtime_reason")["duration_minutes"].sum().reset_index()
                fig_dt_reason = px.pie(
                    reason_counts, values="duration_minutes", names="downtime_reason",
                    title="Downtime by Reason (minutes)",
                    template="plotly_dark", hole=0.4,
                )
                fig_dt_reason.update_layout(
                    height=320, margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_dt_reason, use_container_width=True)

        with col_dt2:
            if "station_id" in downtime_df.columns:
                station_dt = downtime_df.groupby("station_id")["duration_minutes"].sum().reset_index()
                fig_dt_station = px.bar(
                    station_dt.sort_values("duration_minutes", ascending=True),
                    x="duration_minutes", y="station_id", orientation="h",
                    title="Downtime by Station (minutes)",
                    template="plotly_dark",
                    color="duration_minutes", color_continuous_scale="Reds",
                )
                fig_dt_station.update_layout(
                    height=320, margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_dt_station, use_container_width=True)
