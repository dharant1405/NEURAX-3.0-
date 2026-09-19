"""
dashboard/pages/page_root_cause.py

Root-Cause Dashboard page.
Defect × Batch/Station heatmaps, process drift timelines,
ranked hypotheses with confidence and confounders, SHAP importance.
All values computed from data — no hardcoded hypotheses.
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
    st.markdown("# 🔎 Root-Cause Analysis")
    st.markdown(
        "<div class='advisory-banner'>⚠️ All root-cause outputs are statistical hypotheses, "
        "not proven causes. Evidence is correlational. Process investigation required "
        "before taking action. [ADVISORY]</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.get("data_loaded"):
        st.info("👈 Load data first on the **Data Upload** page.")
        return

    if not st.session_state.get("rootcause_run"):
        if st.button("▶ Run Root-Cause Engine", use_container_width=True, type="primary"):
            with st.spinner("Analyzing defect-process associations..."):
                engine.run_rootcause()
            st.rerun()
        st.info("Run the Root-Cause Engine to see hypotheses.")
        return

    hypotheses = st.session_state.get("hypotheses", [])
    inspection_df = st.session_state.get("inspection_df")
    process_df = st.session_state.get("process_parameters_df")
    shap_result = st.session_state.get("shap_result")

    if inspection_df is None:
        st.warning("No inspection data loaded.")
        return

    # ── Defect × Batch heatmap ─────────────────────────────────────────────
    st.markdown("### 🗺️ Defect Family × Batch Heatmap")
    fam_col = None
    for c in ["defect_family"]:
        if c in inspection_df.columns:
            fam_col = c
            break

    if fam_col and "batch_id" in inspection_df.columns:
        defective = inspection_df[
            (inspection_df.get("inspection_result", pd.Series()) == "defective") &
            inspection_df[fam_col].notna() &
            (inspection_df[fam_col] != "None")
        ]
        if len(defective) > 0:
            pivot = defective.groupby(["batch_id", fam_col]).size().unstack(fill_value=0)
            fig_heatmap = px.imshow(
                pivot,
                text_auto=True, aspect="auto",
                color_continuous_scale="Reds",
                template="plotly_dark",
                labels=dict(x="Defect Family", y="Batch", color="Count"),
                title="Defect Counts by Batch × Defect Family",
            )
            fig_heatmap.update_layout(
                height=400, margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── Defect × Station heatmap ───────────────────────────────────────────
    if fam_col and "station_id" in inspection_df.columns:
        defective = inspection_df[
            (inspection_df.get("inspection_result", pd.Series()) == "defective") &
            inspection_df[fam_col].notna() &
            (inspection_df[fam_col] != "None")
        ]
        if len(defective) > 0:
            pivot_station = defective.groupby(["station_id", fam_col]).size().unstack(fill_value=0)
            fig_station_hm = px.imshow(
                pivot_station,
                text_auto=True, aspect="auto",
                color_continuous_scale="Oranges",
                template="plotly_dark",
                labels=dict(x="Defect Family", y="Station", color="Count"),
                title="Defect Counts by Station × Defect Family",
            )
            fig_station_hm.update_layout(
                height=300, margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_station_hm, use_container_width=True)

    # ── Process drift timelines ────────────────────────────────────────────
    if process_df is not None and "batch_id" in process_df.columns:
        st.divider()
        st.markdown("### 📉 Process Parameter Drift by Batch")
        st.markdown("*Computed from data. Dashed vertical line shows first significant change-point.*")

        numeric_params = [c for c in ["temperature", "pressure", "vibration",
                                       "machine_speed", "cycle_time_sec", "tool_wear_percent"]
                          if c in process_df.columns]

        if numeric_params:
            selected_param = st.selectbox(
                "Select parameter to visualize",
                numeric_params,
                index=0,
                key="rc_param_select"
            )
            if "station_id" in process_df.columns:
                stations = sorted(process_df["station_id"].dropna().unique().tolist())
                selected_station = st.selectbox("Station", ["All"] + stations, key="rc_station_select")
            else:
                selected_station = "All"

            param_df = process_df.copy()
            if selected_station != "All" and "station_id" in param_df.columns:
                param_df = param_df[param_df["station_id"] == selected_station]

            batch_param = param_df.groupby("batch_id")[selected_param].agg(["mean", "std"]).reset_index()
            batch_param.columns = ["batch_id", "mean", "std"]

            fig_drift = go.Figure()
            fig_drift.add_trace(go.Scatter(
                x=batch_param["batch_id"],
                y=batch_param["mean"],
                name=f"{selected_param} (mean)",
                line=dict(color="#00BCD4", width=2),
                mode="lines+markers",
            ))
            fig_drift.add_trace(go.Scatter(
                x=batch_param["batch_id"].tolist() + batch_param["batch_id"].tolist()[::-1],
                y=(batch_param["mean"] + batch_param["std"]).tolist() +
                  (batch_param["mean"] - batch_param["std"]).tolist()[::-1],
                fill="toself",
                fillcolor="rgba(0,188,212,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="±1 std",
                showlegend=True,
            ))

            # Simple change-point indicator: where value exceeds 1.5 std from overall mean
            overall_mean = batch_param["mean"].mean()
            overall_std = batch_param["mean"].std()
            change_points = batch_param[
                (batch_param["mean"] - overall_mean).abs() > 1.5 * overall_std
            ]["batch_id"].tolist()
            if change_points:
                fig_drift.add_vline(
                    x=change_points[0], line_dash="dash", line_color="#FF6D00",
                    annotation_text="Change-point", annotation_position="top right",
                )

            fig_drift.update_layout(
                template="plotly_dark", height=320,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Batch", yaxis_title=selected_param,
            )
            st.plotly_chart(fig_drift, use_container_width=True)

    # ── Ranked hypotheses ─────────────────────────────────────────────────
    st.divider()
    st.markdown(f"### 🧠 Root-Cause Hypotheses ({len(hypotheses)} found)")

    if not hypotheses:
        st.info("No hypotheses generated. Ensure inspection and process data are loaded.")
        return

    for hyp in hypotheses[:st.session_state.get("max_hypotheses", 8)]:
        confidence = hyp.get("confidence", 0)
        color = "#00C853" if confidence > 0.6 else ("#FF6D00" if confidence > 0.35 else "#546e7a")

        with st.expander(
            f"[#{hyp.get('rank','?')}] {hyp.get('defect_family','?')} — "
            f"Confidence: {confidence:.0%}",
            expanded=(hyp.get("rank", 99) <= 2),
        ):
            st.markdown(f"""
            <div style='border-left: 4px solid {color}; padding-left: 12px;'>
            <b>Hypothesis:</b> {hyp.get('hypothesis', '—')}
            </div>
            """, unsafe_allow_html=True)

            col_e, col_c = st.columns([2, 1])

            with col_e:
                st.markdown("**Evidence:**")
                for ev in hyp.get("evidence", []):
                    st.markdown(f"• {ev}")

            with col_c:
                st.metric("Confidence", f"{confidence:.1%}")
                effect = hyp.get("effect_size")
                if effect is not None:
                    st.metric("Effect Size", f"{effect:.3f}")
                p_val = hyp.get("adjusted_p_value")
                if p_val is not None:
                    st.metric("Adj. p-value", f"{p_val:.4f}")

            if hyp.get("alternative_explanations"):
                st.markdown("**Alternative Explanations:**")
                for alt in hyp["alternative_explanations"]:
                    st.markdown(f"• ⚖️ *{alt}*")

            if hyp.get("confounders"):
                st.markdown(
                    f"**Potential Confounders:** {', '.join(hyp['confounders'])}"
                )

    # Expand count control
    max_h = st.session_state.get("max_hypotheses", 8)
    if len(hypotheses) > max_h:
        if st.button(f"Show {min(max_h + 5, len(hypotheses)) - max_h} more hypotheses"):
            st.session_state["max_hypotheses"] = max_h + 5
            st.rerun()

    # ── SHAP feature importance ────────────────────────────────────────────
    if shap_result and "feature_importances" in shap_result:
        st.divider()
        st.markdown("### 🎯 Process Feature Importance (SHAP)")
        st.markdown(
            f"*Gradient-boosted model predicting **{shap_result.get('defect_family', 'defect')}** "
            f"occurrence from process parameters. "
            f"Model accuracy: {shap_result.get('model_accuracy', 0):.1%}*"
        )

        shap_df = pd.DataFrame(shap_result["feature_importances"])
        fig_shap = px.bar(
            shap_df.sort_values("mean_shap"),
            x="mean_shap", y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={
                "increases_defect": "#D50000",
                "decreases_defect": "#00C853",
            },
            template="plotly_dark",
            labels={"mean_shap": "Mean |SHAP|", "feature": "Process Parameter"},
        )
        fig_shap.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("Red = higher value increases defect risk. Green = higher value reduces risk. Advisory only.")
