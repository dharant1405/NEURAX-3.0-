"""
dashboard/app.py

InspectIQ — Main Streamlit Application
Defect-to-Profit Decision Support for Multi-Stage Manufacturing

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import logging
import pandas as pd
import streamlit as st
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
dash_cfg_path = _ROOT / "configs" / "dashboard.yaml"
with open(dash_cfg_path) as fh:
    DASH_CFG = yaml.safe_load(fh)

st.set_page_config(
    page_title=DASH_CFG["app"]["title"],
    page_icon=DASH_CFG["app"]["page_icon"],
    layout=DASH_CFG["app"]["layout"],
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1520 50%, #0a0e1a 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1829 0%, #0a1020 100%);
    border-right: 1px solid rgba(0,188,212,0.15);
}
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,188,212,0.2);
    border-radius: 12px;
    padding: 12px;
    backdrop-filter: blur(10px);
}
h1 { font-size: 2rem !important; font-weight: 700 !important; color: #e0f7fa !important; }
h2 { font-size: 1.4rem !important; font-weight: 600 !important; color: #b2ebf2 !important; }
h3 { font-size: 1.1rem !important; font-weight: 500 !important; color: #80deea !important; }

.stButton > button {
    background: linear-gradient(135deg, #006064, #00838f);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00838f, #0097a7);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(0,188,212,0.3);
}
.badge-accept { background: #00C853; color: #000; padding: 2px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85em; }
.badge-reject { background: #D50000; color: #fff; padding: 2px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85em; }
.badge-review { background: #FF6D00; color: #fff; padding: 2px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85em; }
.badge-novel  { background: #AA00FF; color: #fff; padding: 2px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85em; }
.advisory-banner {
    background: linear-gradient(90deg, rgba(255,160,0,0.15), rgba(255,160,0,0.05));
    border: 1px solid rgba(255,160,0,0.4); border-radius: 8px;
    padding: 8px 16px; color: #FFA000; font-size: 0.85em;
    font-weight: 500; margin-bottom: 12px;
}
.assumption-badge {
    background: rgba(255,160,0,0.15); border: 1px solid rgba(255,160,0,0.4);
    border-radius: 4px; padding: 2px 6px; color: #FFA000;
    font-size: 0.75em; font-style: italic;
}
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
[data-testid="stExpander"] { border: 1px solid rgba(0,188,212,0.15); border-radius: 8px; }
.stProgress > div > div { background: linear-gradient(90deg, #006064, #00BCD4); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize session state
# ─────────────────────────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "data_loaded": False,
        "vision_run": False,
        "rootcause_run": False,
        "flow_run": False,
        "economics_run": False,
        "current_page": "Data Upload",
        "inspection_df": None,
        "products_df": None,
        "process_parameters_df": None,
        "production_df": None,
        "downtime_df": None,
        "economics_df": None,
        "stations_df": None,
        "batches_df": None,
        "defect_catalog_df": None,
        "vision_results": None,
        "hypotheses": [],
        "bottleneck_results": [],
        "flow_losses": {},
        "baseline_simulation": None,
        "scenario_simulation": None,
        "scenario_delta": {},
        "economics_summary": {},
        "recommendations": [],
        "quality_reports": {},
        "decision_summary": {},
        "shap_result": None,
        "threshold_optimization": None,
        "max_hypotheses": 8,
        "filter_variants": [],
        "filter_batches": [],
        "filter_stations": [],
        "filter_defect_families": [],
        "filter_shifts": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# ─────────────────────────────────────────────────────────────────────────────
# Global filter helper
# ─────────────────────────────────────────────────────────────────────────────
def _apply_filters():
    """Render filter widgets populated dynamically from data — never hardcoded."""
    # Variants
    for df_key, col in [("inspection_df", "variant_id"), ("products_df", "product_variant")]:
        df = st.session_state.get(df_key)
        if df is not None and col in df.columns:
            variants = sorted(df[col].dropna().unique().tolist())
            if variants:
                st.multiselect("Product Variants", variants, key="filter_variants",
                               default=st.session_state.get("filter_variants", []),
                               placeholder="All variants")
            break

    # Batches
    for df_key, col in [("inspection_df", "batch_id"), ("products_df", "batch_id")]:
        df = st.session_state.get(df_key)
        if df is not None and col in df.columns:
            batches = sorted(df[col].dropna().unique().tolist())
            if batches:
                st.multiselect("Batches", batches[:50], key="filter_batches",
                               default=st.session_state.get("filter_batches", []),
                               placeholder="All batches")
            break

    # Stations
    for df_key, col in [("production_df", "station_id"), ("stations_df", "station_id")]:
        df = st.session_state.get(df_key)
        if df is not None and col in df.columns:
            stations = sorted(df[col].dropna().unique().tolist())
            if stations:
                st.multiselect("Stations", stations, key="filter_stations",
                               default=st.session_state.get("filter_stations", []),
                               placeholder="All stations")
            break

    # Defect families
    vision_df = st.session_state.get("vision_results") or st.session_state.get("inspection_df")
    if vision_df is not None:
        for col in ["defect_family", "defect_family_predicted"]:
            if col in vision_df.columns:
                families = sorted([
                    f for f in vision_df[col].dropna().unique().tolist()
                    if f and f not in ("None", "")
                ])
                if families:
                    st.multiselect("Defect Families", families, key="filter_defect_families",
                                   default=st.session_state.get("filter_defect_families", []),
                                   placeholder="All families")
                break


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <div style='font-size: 2.5rem; margin-bottom: 0.3rem;'>🔬</div>
        <div style='font-size: 1.3rem; font-weight: 700; color: #e0f7fa;'>InspectIQ</div>
        <div style='font-size: 0.75rem; color: #4dd0e1; letter-spacing: 0.1em;'>DEFECT-TO-PROFIT PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    data_ok = st.session_state["data_loaded"]

    def _dot(ok: bool) -> str:
        return "🟢" if ok else "⚪"

    st.markdown(f"""
**System Status**
{_dot(data_ok)} Data Loaded
{_dot(st.session_state['vision_run'])} Vision Engine
{_dot(st.session_state['rootcause_run'])} Root-Cause Engine
{_dot(st.session_state['flow_run'])} Flow Engine
{_dot(st.session_state['economics_run'])} Economics Engine
    """)

    st.divider()

    pages = {
        "📁 Data Upload": "Data Upload",
        "🔍 Quality": "Quality",
        "🔎 Root Cause": "Root Cause",
        "⚙️ Flow": "Flow",
        "💰 Profit & What-If": "Profit & What-If",
    }

    for label, page_key in pages.items():
        if st.button(label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state["current_page"] = page_key
            st.rerun()

    st.divider()

    if data_ok:
        st.markdown("### 🔽 Global Filters")
        _apply_filters()

    st.divider()
    st.markdown(
        "<div style='font-size:0.7em; color: #546e7a; text-align: center;'>"
        "NEURAX Hackathon 3.0 · Domain 2<br>AI in Industry and Automation"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page routing
# ─────────────────────────────────────────────────────────────────────────────
current_page = st.session_state.get("current_page", "Data Upload")

# Lazy imports to avoid issues at startup
if current_page == "Data Upload":
    from dashboard.pages import page_data_upload
    page_data_upload.render()
elif current_page == "Quality":
    from dashboard.pages import page_quality
    page_quality.render()
elif current_page == "Root Cause":
    from dashboard.pages import page_root_cause
    page_root_cause.render()
elif current_page == "Flow":
    from dashboard.pages import page_flow
    page_flow.render()
elif current_page == "Profit & What-If":
    from dashboard.pages import page_profit_whatif
    page_profit_whatif.render()
