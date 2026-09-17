import streamlit as st
import pandas as pd
from pathlib import Path

from src.inference import SegmentPredictor
from components.ui_blocks import (
    render_overview,
    render_segments,
    render_anomalies,
    render_customer_lookup,
)

# Page Setup
st.set_page_config(
    page_title="Customer Intelligence Engine",
    page_icon="assets/favicon.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

css_file = Path(__file__).parent / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

# Initialize Engine with Caching
@st.cache_resource
def get_predictor():
    return SegmentPredictor()

@st.cache_data(show_spinner=False)
def run_batch_inference(df: pd.DataFrame):
    engine = get_predictor()
    return engine.process_batch(df)

# Sidebar Branding Header
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-logo">◈</div>
        <div class="sidebar-title-container">
            <span class="sidebar-title">CUSTOMER</span>
            <span class="sidebar-subtitle">INTELLIGENCE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<p class="sidebar-section-label">NAVIGATION</p>', unsafe_allow_html=True)

nav_options = {
    "📊  Overview": "Overview",
    "🧩  Segments": "Segments",
    "⚡  Anomalies": "Anomalies",
    "🔍  Customers": "Customers"
}

selected_label = st.sidebar.radio(
    "Navigation",
    list(nav_options.keys()),
    label_visibility="collapsed"
)
navigation = nav_options[selected_label]

st.sidebar.divider()

# System Status Card
st.sidebar.markdown(
    """
    <div class="sidebar-info-card">
        <div class="info-card-header">
            <span class="status-dot"></span> Model Status
        </div>
        <p class="info-card-text">
            <b>Engine:</b> Unsupervised ML v1.0<br>
            <b>Model:</b> K-Means + Isolation Forest<br>
            <b>Source:</b> Online Retail II
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Data Loading Logic
@st.cache_data
def load_fallback_data():
    """Fallback loader checking anomalies export first, then standard processed CSV."""
    processed_dir = Path(__file__).resolve().parent / "data" / "processed"
    
    segmentation_path = processed_dir / "final_customer_segmentation.csv"
    anomalies_path = processed_dir / "customer_anomalies.csv"
    
    segmentation_df = pd.read_csv(segmentation_path, index_col='Customer ID')
    anomalies_df = pd.read_csv(anomalies_path, index_col='Customer ID')

    return segmentation_df, anomalies_df

segmentation_data, MultiMethod_Anomaly = load_fallback_data()

with st.spinner("Processing customer profiles..."):
    clean_input = segmentation_data.drop('Cluster', axis=1)
    results_df = run_batch_inference(clean_input)
    
    # Identify pure behavioral feature columns
    feature_cols = [c for c in clean_input.columns]

# route Destinations
if navigation == "Overview":
    render_overview(results_df, MultiMethod_Anomaly, feature_cols)
elif navigation == "Segments":
    render_segments(results_df, feature_cols)
elif navigation == "Anomalies":
    render_anomalies(results_df, MultiMethod_Anomaly, feature_cols)
elif navigation == "Customers":
        render_customer_lookup(results_df, feature_cols)

# Footer
st.divider()
st.markdown(
    """
    <div class="custom-footer">
        <p>
            Built with <b>Streamlit</b>, <b>Scikit-learn</b> & <b>Plotly</b> | 
            Data source: <b>Online Retail II</b> transaction dataset.
        </p>
        <p>
            Developed by <b>Mazen Mahmoud</b> • 
            <a href="https://github.com/MazenSr" target="_blank">GitHub</a> • 
            <a href="https://www.linkedin.com/in/mazen-mahmoud-ds/" target="_blank">LinkedIn</a> • 
            <a href="mailto:mazen.mahmoud420409@gmail.com">Contact Me</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)        