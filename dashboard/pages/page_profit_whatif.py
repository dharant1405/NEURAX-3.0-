"""
dashboard/pages/page_profit_whatif.py

Profit & What-If Dashboard page.
Margin trends, cost breakdown, economics-optimized threshold curve,
scenario sliders, delta metrics, advisory recommendations.
All SIMULATED / ADVISORY labelling throughout.
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
ADVISORY = "SIMULATED / ADVISORY"


@st.cache_resource
def get_engine() -> EngineRunner:
    return EngineRunner()


def render():
    engine = get_engine()
    st.markdown("# 💰 Profit & What-If Analysis")

    st.markdown(
        "<div class='advisory-banner'>⚠️ All profit estimates, scenario results, and recommendations "
        "are SIMULATED / ADVISORY. They are based on data-calibrated simulation and should not be "
        "acted on without further investigation and expert review.</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.get("data_loaded"):
        st.info("👈 Load data first on the **Data Upload** page.")
        return

    # ── Run economics if needed ────────────────────────────────────────────
    if not st.session_state.get("economics_run"):
        if st.button("▶ Run Economics Engine", use_container_width=True, type="primary"):
            with st.spinner("Computing economics..."):
                engine.run_economics()
                if not st.session_state.get("recommendations"):
                    engine.run_recommendations()
            st.rerun()
        st.info("Run the Economics Engine to see profit analysis.")
        return

    economics_summary = st.session_state.get("economics_summary", {})
    econ_df = st.session_state.get("economics_df")
    recommendations = st.session_state.get("recommendations", [])
    threshold_opt = st.session_state.get("threshold_optimization", {})

    summary = economics_summary.get("summary", {})
    currency = summary.get("currency", "₹")

    # ── KPI Row ────────────────────────────────────────────────────────────
    st.markdown("### 💵 Economic Summary")
    col1, col2, col3, col4 = st.columns(4)

    total_revenue = summary.get("total_revenue", 0)
    total_cost = summary.get("total_cost", 0)
    total_profit = summary.get("total_profit", 0)
    avg_margin = summary.get("avg_margin_pct", 0)

    col1.metric("Total Revenue", f"{currency}{total_revenue:,.0f}")
    col2.metric("Total Cost", f"{currency}{total_cost:,.0f}")
    col3.metric("Total Profit", f"{currency}{total_profit:,.0f}",
                f"{'+' if total_profit >= 0 else ''}{total_profit/max(total_revenue,1)*100:.1f}% margin")
    col4.metric("Avg Margin %", f"{avg_margin:.1f}%")

    # Assumption badge if fallback values used
    if economics_summary.get("assumptions_used"):
        assumed = economics_summary.get("assumed_fields", [])
        st.markdown(
            f"<div class='assumption-badge'>⚠️ ASSUMPTION: Fields estimated from config: "
            f"{', '.join(assumed[:5])}. Data values take precedence when available.</div>",
            unsafe_allow_html=True,
        )

    # ── Cost breakdown ────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("### 🥧 Cost Breakdown")
        cost_breakdown = economics_summary.get("cost_breakdown", {})
        if cost_breakdown:
            cb_df = pd.DataFrame(
                [{"Component": k.replace("_", " ").title(), "Cost": v}
                 for k, v in cost_breakdown.items() if v > 0]
            )
            fig_cost = px.pie(
                cb_df, values="Cost", names="Component", hole=0.4,
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_cost.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cost, use_container_width=True)

    with col_b:
        st.markdown("### 📊 Profit by Variant")
        per_variant = economics_summary.get("per_variant", {})
        if per_variant:
            pv_df = pd.DataFrame([
                {"Variant": k, "Avg Profit": v["avg_profit"], "Avg Margin %": v["avg_margin_pct"]}
                for k, v in per_variant.items()
            ])
            fig_variant = px.bar(
                pv_df, x="Variant", y="Avg Profit",
                color="Avg Margin %", color_continuous_scale="RdYlGn",
                template="plotly_dark",
                labels={"Avg Profit": f"Avg Profit ({currency})", "Variant": "Product Variant"},
            )
            fig_variant.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_variant, use_container_width=True)

    # ── Profit trend over batches ──────────────────────────────────────────
    st.divider()
    st.markdown("### 📈 Profit Trend by Batch")

    if econ_df is not None and "batch_id" in econ_df.columns and "profit" in econ_df.columns:
        batch_profit = econ_df.groupby("batch_id")["profit"].agg(["mean", "std"]).reset_index()
        batch_profit.columns = ["batch_id", "mean_profit", "std_profit"]

        fig_profit = go.Figure()
        fig_profit.add_trace(go.Scatter(
            x=batch_profit["batch_id"], y=batch_profit["mean_profit"],
            name="Mean Profit / Unit", line=dict(color="#00C853", width=2),
            mode="lines+markers",
        ))
        # Confidence band
        fig_profit.add_trace(go.Scatter(
            x=batch_profit["batch_id"].tolist() + batch_profit["batch_id"].tolist()[::-1],
            y=(batch_profit["mean_profit"] + batch_profit["std_profit"]).tolist() +
              (batch_profit["mean_profit"] - batch_profit["std_profit"]).tolist()[::-1],
            fill="toself", fillcolor="rgba(0,200,83,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="±1 std",
        ))
        fig_profit.update_layout(
            template="plotly_dark", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Batch", yaxis_title=f"Profit per Unit ({currency})",
        )
        st.plotly_chart(fig_profit, use_container_width=True)

    # ── Cost-optimal threshold curve ───────────────────────────────────────
    if threshold_opt and "cost_curve" in threshold_opt:
        st.divider()
        st.markdown("### ⚖️ Economics-Optimal Inspection Threshold")
        st.markdown(
            "<div class='advisory-banner'>Threshold optimized to minimize expected cost "
            f"(False Accept Cost: {currency}{threshold_opt.get('false_accept_cost_used', 0):.0f} | "
            f"False Reject Cost: {currency}{threshold_opt.get('false_reject_cost_used', 0):.0f} | "
            f"Review Cost: {currency}{threshold_opt.get('review_cost_used', 0):.0f}). "
            f"[{ADVISORY}]</div>",
            unsafe_allow_html=True,
        )

        cc_df = pd.DataFrame(threshold_opt["cost_curve"])
        fig_thresh = go.Figure()
        fig_thresh.add_trace(go.Scatter(
            x=cc_df["threshold"], y=cc_df["expected_cost"],
            name="Expected Cost", line=dict(color="#FF6D00"),
        ))
        fig_thresh.add_trace(go.Scatter(
            x=cc_df["threshold"], y=cc_df["false_accept_rate"] * max(cc_df["expected_cost"]),
            name="False Accept Rate (scaled)", line=dict(color="#D50000", dash="dot"),
        ))
        fig_thresh.add_trace(go.Scatter(
            x=cc_df["threshold"], y=cc_df["false_reject_rate"] * max(cc_df["expected_cost"]),
            name="False Reject Rate (scaled)", line=dict(color="#FF6D00", dash="dot"),
        ))
        opt_thresh = threshold_opt.get("optimal_high_threshold", 0.5)
        fig_thresh.add_vline(x=opt_thresh, line_dash="dash", line_color="#00C853",
                             annotation_text=f"Optimal: {opt_thresh:.3f}")
        fig_thresh.update_layout(
            template="plotly_dark", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Reject Threshold", yaxis_title="Expected Cost",
        )
        st.plotly_chart(fig_thresh, use_container_width=True)
        col_th1, col_th2 = st.columns(2)
        col_th1.metric("Optimal Reject Threshold",
                       f"{threshold_opt.get('optimal_high_threshold', 0):.3f}",
                       "[ADVISORY]")
        col_th2.metric("Expected Cost at Optimal",
                       f"{currency}{threshold_opt.get('expected_cost_at_optimal', 0):,.0f}",
                       "[ADVISORY]")

    # ── What-If Scenario Simulator ─────────────────────────────────────────
    st.divider()
    st.markdown("### 🎛️ What-If Scenario Simulator")
    st.markdown(
        f"<div class='advisory-banner'>All scenario results are {ADVISORY}. "
        "They are simulation-based estimates, not predictions of actual outcomes.</div>",
        unsafe_allow_html=True,
    )

    with st.form("scenario_form"):
        st.markdown("**Adjust parameters relative to baseline:**")
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            defect_mult = st.slider("Defect Rate Multiplier", 0.1, 3.0, 1.0, 0.05,
                                    help="1.0 = baseline defect rate")
        with col_s2:
            ct_mult = st.slider("Cycle Time Multiplier", 0.5, 2.0, 1.0, 0.05,
                                help="1.0 = baseline cycle time")
        with col_s3:
            dt_mult = st.slider("Downtime Multiplier", 0.0, 3.0, 1.0, 0.1,
                                help="1.0 = baseline downtime")

        run_scenario = st.form_submit_button("▶ Run Scenario [SIMULATED / ADVISORY]",
                                              use_container_width=True)

    if run_scenario:
        with st.spinner("Running scenario simulation..."):
            scenario_params = {
                "defect_rate_multiplier": defect_mult,
                "cycle_time_multiplier": ct_mult,
                "downtime_multiplier": dt_mult,
            }
            result = engine.run_scenario(scenario_params)
            st.success(f"✓ Scenario simulation complete. [{ADVISORY}]")

    # Show scenario results
    baseline = st.session_state.get("baseline_simulation")
    scenario = st.session_state.get("scenario_simulation")
    delta = st.session_state.get("scenario_delta", {})

    if baseline and scenario:
        st.markdown("#### 📊 Scenario vs Baseline Comparison")
        st.markdown(
            f"<div class='advisory-banner'>All values below are {ADVISORY}. "
            "Simulation uses calibrated station parameters from data.</div>",
            unsafe_allow_html=True,
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("**Baseline**")
            st.metric("Throughput/hr", f"{baseline.get('throughput_per_hour', 0):.1f} units")
            st.metric("Scrap Units", f"{baseline.get('scrap_units', 0):.0f}")
            st.metric("Rework Units", f"{baseline.get('rework_units', 0):.0f}")

        with col_b2:
            st.markdown(f"**Scenario** [{ADVISORY}]")
            d_tput = delta.get("delta_throughput_per_hour", 0)
            d_margin = delta.get("delta_estimated_margin", 0)

            st.metric("Throughput/hr",
                      f"{scenario.get('throughput_per_hour', 0):.1f} units",
                      f"{'+' if d_tput >= 0 else ''}{d_tput:.1f} units/hr [{ADVISORY}]")
            st.metric("Scrap Units",
                      f"{scenario.get('scrap_units', 0):.0f}",
                      f"{delta.get('delta_scrap_units', 0):+.0f} [{ADVISORY}]")
            st.metric("Est. Margin Δ/shift",
                      f"{currency}{delta.get('delta_estimated_margin', 0):+,.0f}",
                      f"[{ADVISORY}]")

        if delta.get("assumptions"):
            with st.expander("📋 Assumptions used in this scenario"):
                for a in delta["assumptions"]:
                    st.markdown(f"• *{a}*")

    # ── Recommendations ───────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 💡 Advisory Recommendations [{ADVISORY}]")

    if not recommendations:
        if st.button("Generate Recommendations", use_container_width=True):
            with st.spinner("Generating recommendations..."):
                engine.run_recommendations()
            st.rerun()

    for rec in recommendations:
        confidence = rec.get("confidence", 0)
        color = "#00C853" if confidence > 0.6 else ("#FF6D00" if confidence > 0.35 else "#546e7a")

        with st.expander(
            f"[#{rec.get('rank', '?')}] {rec.get('action', '')[:80]}... "
            f"(Confidence: {confidence:.0%})",
            expanded=(rec.get("rank", 99) <= 2),
        ):
            st.markdown(
                f"<div class='advisory-banner'>{rec.get('status', ADVISORY)}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(f"**Action:** {rec.get('action', '—')}")

            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                st.markdown("**Evidence:**")
                for ev in rec.get("evidence", []):
                    st.markdown(f"• {ev}")

            with col_r2:
                st.metric("Confidence", f"{confidence:.0%}")
                impact = rec.get("expected_impact", {})
                if "description" in impact:
                    st.caption(impact["description"])

            if rec.get("assumptions"):
                st.markdown("**Assumptions:**")
                for a in rec["assumptions"]:
                    st.markdown(f"• *{a}*")

            if rec.get("alternative_explanations"):
                st.markdown("**Alternative Explanations:**")
                for alt in rec["alternative_explanations"]:
                    st.markdown(f"• ⚖️ *{alt}*")
