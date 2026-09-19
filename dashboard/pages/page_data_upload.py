"""
dashboard/pages/page_data_upload.py

Data Upload & Validation page.
Handles synthetic data generation, file upload, schema mapping preview,
and data quality report display.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
from io import StringIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.engine_runner import EngineRunner

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[2]
_SYNTHETIC_DIR = _ROOT / "data" / "synthetic"


@st.cache_resource
def get_engine() -> EngineRunner:
    return EngineRunner()


def render():
    engine = get_engine()

    st.markdown("# 📁 Data Upload & Validation")
    st.markdown(
        "Load your manufacturing dataset. The system adapts to your column names "
        "via the schema mapping in `configs/schema_mapping.yaml`."
    )

    # ── Data source selection ──────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Option 1 — Use Synthetic Demo Dataset")
        st.markdown(
            "Generate a realistic synthetic dataset with known defect patterns, "
            "process drift, and a bottleneck station."
        )

        with st.expander("ℹ️ What patterns are in the synthetic data?"):
            st.markdown("""
            **Injected patterns (discoverable by the AI engines):**
            - 🌡️ **Pattern 1**: Station S3 develops a temperature drift in later batches → Crack + Discoloration defects increase
            - 📳 **Pattern 2**: Station S2 vibration increases due to tool wear → Scratch + Surface_Pit defects increase
            - ⚙️ **Pattern 3**: Station S4 has the longest cycle time → production bottleneck
            - 📦 **Pattern 4**: Batches B009–B011 have a material quality issue → Porosity defect spike
            - 🔍 **Pattern 5**: A small fraction of units show novel defects (unusual process conditions)
            - 📉 **Pattern 6**: Batch-to-batch process drift in later batches
            - 🏷️ **Pattern 7**: Different product variants have different normal operating ranges
            """)

        n_units = st.slider("Number of units", 500, 15000, 5000, 500,
                            help="More units = more realistic analysis but slower loading")
        n_batches = st.slider("Number of batches", 5, 25, 10, 1)

        if st.button("⚡ Generate & Load Synthetic Dataset", use_container_width=True):
            with st.spinner("Generating synthetic dataset..."):
                try:
                    from data.synthetic.generator import SyntheticDataGenerator, DEFAULT_CONFIG
                    cfg = DEFAULT_CONFIG.copy()
                    cfg["n_units"] = n_units
                    cfg["n_batches"] = n_batches
                    _SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
                    gen = SyntheticDataGenerator(cfg)
                    gen.generate_all(_SYNTHETIC_DIR)
                    st.success(f"✓ Generated {n_units} units across {n_batches} batches")
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")
                    return

            with st.spinner("Loading and validating dataset..."):
                try:
                    result = engine.load_from_directory(_SYNTHETIC_DIR)
                    tables_loaded = result.get("tables", [])
                    st.success(f"✓ Loaded {len(tables_loaded)} tables: {', '.join(tables_loaded)}")
                except Exception as exc:
                    st.error(f"Loading failed: {exc}")
                    return

    with col2:
        st.markdown("### Option 2 — Upload Your Own Dataset")
        st.markdown("Upload CSV files. Column names are mapped via `schema_mapping.yaml`.")

        uploaded_files = st.file_uploader(
            "Upload dataset files (CSV)",
            accept_multiple_files=True,
            type=["csv", "parquet", "json"],
            help="Supported: inspection, products, process_parameters, production, "
                 "downtime, economics, stations, batches, defect_catalog"
        )

        if uploaded_files and st.button("📤 Load Uploaded Files", use_container_width=True):
            with st.spinner("Loading files..."):
                result = engine.load_uploaded_files(uploaded_files)
                tables_loaded = result.get("tables", [])
                if tables_loaded:
                    st.success(f"✓ Loaded: {', '.join(tables_loaded)}")
                else:
                    st.warning("No tables could be loaded. Check file names and format.")

    # ── Loaded data preview ────────────────────────────────────────────────
    if st.session_state.get("data_loaded"):
        st.divider()
        st.markdown("## 📊 Loaded Dataset Overview")

        # Summary metrics
        tables_in_session = {
            k: v for k, v in {
                "Inspection": st.session_state.get("inspection_df"),
                "Products": st.session_state.get("products_df"),
                "Process Params": st.session_state.get("process_parameters_df"),
                "Production": st.session_state.get("production_df"),
                "Economics": st.session_state.get("economics_df"),
                "Stations": st.session_state.get("stations_df"),
                "Batches": st.session_state.get("batches_df"),
                "Downtime": st.session_state.get("downtime_df"),
            }.items() if v is not None
        }

        cols = st.columns(min(len(tables_in_session), 4))
        for i, (name, df) in enumerate(tables_in_session.items()):
            with cols[i % 4]:
                st.metric(name, f"{len(df):,} rows", f"{len(df.columns)} cols")

        # Dynamic dataset discovery summary
        st.divider()
        col3, col4, col5 = st.columns(3)

        insp_df = st.session_state.get("inspection_df")
        prod_df = st.session_state.get("production_df")
        econ_df = st.session_state.get("economics_df")

        with col3:
            if insp_df is not None:
                st.markdown("**Discovered from data:**")
                # Variants (dynamically discovered)
                for col in ["variant_id", "product_variant"]:
                    if col in insp_df.columns:
                        variants = insp_df[col].dropna().unique().tolist()
                        st.write(f"• **{len(variants)} product variants**: {', '.join(str(v) for v in variants[:5])}")
                        break
                # Defect families
                for col in ["defect_family"]:
                    if col in insp_df.columns:
                        families = [f for f in insp_df[col].dropna().unique().tolist() if f and f != "None"]
                        st.write(f"• **{len(families)} defect families**: {', '.join(str(f) for f in families[:5])}")
                        break
                # Batches
                if "batch_id" in insp_df.columns:
                    n_batches_found = insp_df["batch_id"].nunique()
                    st.write(f"• **{n_batches_found} batches** discovered")

        with col4:
            if prod_df is not None:
                st.markdown("**Production stations:**")
                if "station_id" in prod_df.columns:
                    stations = prod_df["station_id"].dropna().unique().tolist()
                    for sid in sorted(stations):
                        st.write(f"• {sid}")

        with col5:
            if econ_df is not None:
                st.markdown("**Economics coverage:**")
                econ_cols = ["selling_price", "material_cost", "profit", "total_cost"]
                for col in econ_cols:
                    if col in econ_df.columns:
                        coverage = econ_df[col].notna().mean() * 100
                        st.write(f"• {col}: {coverage:.0f}% coverage")

        # ── Data quality reports ───────────────────────────────────────────
        reports = st.session_state.get("quality_reports", {})
        if reports:
            st.divider()
            st.markdown("## ✅ Data Quality Reports")

            for table_key, report in reports.items():
                status = "✅" if report.is_valid else "⚠️"
                with st.expander(f"{status} {table_key.replace('_', ' ').title()} — {report.total_records:,} records"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Records", f"{report.total_records:,}")
                    c2.metric("Valid Records", f"{report.valid_records:,}")
                    c3.metric("Duplicates", f"{report.duplicate_records:,}")

                    if report.errors:
                        st.error("**Errors:**\n" + "\n".join(f"• {e}" for e in report.errors))
                    if report.warnings:
                        st.warning("**Warnings:**\n" + "\n".join(f"• {w}" for w in report.warnings))

                    # Null % chart
                    null_pcts = {k: v for k, v in report.null_pct_by_field.items()
                                 if not k.startswith("_ext_") and v > 0}
                    if null_pcts:
                        null_df = pd.DataFrame(
                            {"field": list(null_pcts.keys()), "null_pct": list(null_pcts.values())}
                        ).sort_values("null_pct", ascending=True)
                        fig = px.bar(
                            null_df, x="null_pct", y="field", orientation="h",
                            title="Null % by Field",
                            color="null_pct", color_continuous_scale="Reds",
                            template="plotly_dark",
                        )
                        fig.update_layout(height=max(150, len(null_pcts) * 25), margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True)

        # ── Run engines section ────────────────────────────────────────────
        st.divider()
        st.markdown("## 🚀 Run Analysis Engines")
        st.markdown(
            "Run all engines in sequence, or navigate to each dashboard page "
            "to run them individually with more control."
        )

        if st.button("▶ Run All Engines (Full Analysis)", use_container_width=True, type="primary"):
            progress = st.progress(0, text="Starting analysis...")

            with st.spinner("Running Vision Engine..."):
                engine.run_vision()
            progress.progress(25, text="Vision Engine ✓")

            with st.spinner("Running Root-Cause Engine..."):
                engine.run_rootcause()
            progress.progress(50, text="Root-Cause Engine ✓")

            with st.spinner("Running Flow Engine..."):
                engine.run_flow()
            progress.progress(70, text="Flow Engine ✓")

            with st.spinner("Running Economics Engine..."):
                engine.run_economics()
            progress.progress(85, text="Economics Engine ✓")

            with st.spinner("Generating Recommendations..."):
                engine.run_recommendations()
            progress.progress(100, text="All engines complete!")

            st.success(
                "✓ Full analysis complete! Navigate to Quality, Root Cause, Flow, "
                "and Profit & What-If pages to explore results."
            )

        # ── Schema mapping display ─────────────────────────────────────────
        with st.expander("🗂️ View Active Schema Mapping"):
            st.markdown(
                "The table below shows how external dataset columns are mapped "
                "to InspectIQ's internal schema. Edit `configs/schema_mapping.yaml` to adapt."
            )
            mapping_path = _ROOT / "configs" / "schema_mapping.yaml"
            with open(mapping_path, encoding="utf-8") as fh:
                st.code(fh.read(), language="yaml")
