from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
from sklearn.decomposition import PCA

CLUSTER_COLORS = {"Cluster 0": "#2b5c8f", "Cluster 1": "#469d89", "Cluster 2": "#d95f02"}
ANOMALY_COLOR = "#d9381e"
NEUTRAL_COLOR = "#2b5c8f"

CLUSTER_PERSONAS = {
    0: "The One-Offs",
    1: "The Casual Weekenders",
    2: "The Loyal High-Rollers",
}
CLUSTER_PROFILES = {
    0: {
        "title": "Single-Purchase / At-Risk",
        "description": "Lowest purchase frequency (1.0), zero customer lifetime, and high recency (284 days). These customers appear to have made a single purchase with no observed repeat activity."
    },
    1: {
        "title": "Weekend Shoppers",
        "description": "Moderate monetary value with a highly dominant WeekendPurchaseRatio (0.78), indicating a strongly weekend-oriented purchasing pattern."
    },
    2: {
        "title": "High-Value Champions",
        "description": "Highest purchase frequency (6.0), lowest recency (43 days), and highest monetary value ($2,286.19), indicating frequent, recent, and higher-value purchasing behavior."
    }
}

def render_header(title: str, repo_url: str = "https://github.com/MazenSr/Customer-Intelligence-Engine"):
    """Renders a simple page header exactly like st.title() with a clean GitHub link."""
    st.markdown(
        f'<div class="app-header-container">'
        f'<h1 class="app-header-title">{title}</h1>'
        f'<a href="{repo_url}" target="_blank" class="github-link">'
        f'<svg viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
        f'View on GitHub'
        f'</a>'
        f'</div>',
        unsafe_allow_html=True
    )

@st.cache_resource
def load_pca_models():
    model_path = Path(__file__).parent.parent / "models" / "preprocessor_robust.joblib"
    preprocessor = joblib.load(model_path)
    return preprocessor

@st.cache_data
def get_pca_coordinates(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Computes 2D PCA projections for visualization purposes only."""
    preprocessor = load_pca_models()
    preprocessed_data = preprocessor.transform(df[feature_cols])
    
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(preprocessed_data)
    
    return pd.DataFrame(components, columns=["PCA1", "PCA2"], index=df.index)


def render_overview(results_df: pd.DataFrame, MultiMethod_Anomaly, feature_cols: list):
    """Page 1: Executive Overview Dashboard."""
    render_header("E-Commerce Customer Intelligence")
    st.caption("Customer Segmentation, Behavioral Analytics & Anomaly Detection")

    # Executive Metrics
    total_customers = len(results_df)
    active_customers = (results_df["Frequency"] > 1).sum()
    n_clusters = results_df["Cluster"].nunique()
    anomalies_customers = results_df["Is_Anomaly"].sum()
    anomalies_customers_pct = (anomalies_customers / total_customers) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{total_customers:,}")
    c2.metric("Repeat Customers", f"{active_customers:,}")
    c3.metric("Segments", f"{n_clusters}")
    c4.metric("Anomalies", f"{anomalies_customers_pct:.1f}%", f"{anomalies_customers} flagged", delta_color="inverse")

    st.divider()

    # Main Visual: 2D PCA Cluster Representation
    st.subheader("Customer Segmentation")
    pca_df = get_pca_coordinates(results_df, feature_cols)
    pca_df["Cluster"] = results_df["Cluster"].astype(str).apply(lambda x: f"Cluster {x}")

    fig = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        color_discrete_map=CLUSTER_COLORS,
        opacity=0.7,
        template="plotly_white",
        title="2D PCA Population Map",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
    col_a, col_b = st.columns([1.8, 1])
    # Customer Segmentation part
    with col_a:
        st.subheader("Customer Segments")
        st.caption("Distribution of customers across behavioral segments")

        seg_counts = (
            results_df.loc[~results_df["Is_Anomaly"]]["Cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )
        seg_counts.columns = ["Cluster", "Customers"]
        seg_counts["Share"] = (seg_counts["Customers"] / seg_counts["Customers"].sum()) * 100
        seg_counts["Cluster_Label"] = seg_counts["Cluster"].astype(str).apply(lambda x: f"Cluster {x}")

        seg_counts["Share_Formatted"] = seg_counts["Share"].map(lambda x: f"{x:.1f}%")
        fig = px.bar(
            seg_counts,
            x="Cluster_Label",
            y="Customers",
            color="Cluster_Label",
            color_discrete_map=CLUSTER_COLORS,
            custom_data=["Share_Formatted"],
            template="plotly_white"
        )

        fig.update_traces(marker=dict(line=dict(width=2, color="#CCCCCC"),cornerradius=6),
            hovertemplate=(
            "<b>%{x}</b><br>"
            "Customers: <b>%{y:,}</b><br>"
            "Share: <b>%{customdata[0]}</b>"
            "<extra></extra>"
        )
    )

        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title=None, showgrid=False,tickfont=dict(size=15)),
            yaxis=dict(title="Customers",range=[0, seg_counts["Customers"].max() * 1.25]),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

    # Behavioral Anomalies part
    with col_b:
        st.subheader("Behavioral Anomalies")
        st.caption("Customers requiring further investigation")

        high_conf_count = MultiMethod_Anomaly['High_Confidence_Anomaly'].sum()
        high_conf_pct = (high_conf_count / anomalies_customers * 100)

        # side-by-side metric cards
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Flagged Customers",
                value=f"{anomalies_customers:,}",
                delta=f"{anomalies_customers_pct:.1f}% of customer base",
                delta_color="inverse",
                help="Total accounts flagged by the Isolation Forest anomaly pipeline."
            )

        with col2:
            st.metric(
                label="High-Confidence Flags",
                value=f"{high_conf_count:,}",
                delta=f"{high_conf_pct:.1f}% of flagged total",
                delta_color="inverse",
                help="Multi-method consensus anomalies requiring immediate manual triage."
            )

        st.markdown(
            """
            <div style="
                padding: 14px 16px;
                border-radius: 10px;
                border: 1px solid rgba(128,128,128,0.25);
                margin-top: 10px;
            ">
                <div style="font-size: 0.85rem; opacity: 0.7;">
                    Detection scope
                </div>
                <div style="font-size: 0.95rem; margin-top: 5px;">
                    Unusual patterns across customer spending,
                    purchase frequency, product activity, and
                    cancellation behavior.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_segments(results_df: pd.DataFrame, feature_cols: list):
    """Page 2: Customer Segment Profiling."""
    render_header("Customer Segment Profiles")
    st.caption("Explore how customer groups differ in purchasing behavior, engagement, and value.")

    inliers = results_df[~results_df["Is_Anomaly"]]

    # Segment Share Progress Bars
    st.subheader("Segment Distribution")
    counts = inliers["Cluster"].value_counts(normalize=True).sort_index()
    for cluster_id, share in counts.items():
        persona = CLUSTER_PERSONAS.get(int(cluster_id),f"Cluster {cluster_id}")
        st.write(f"**Cluster {cluster_id} — {persona}**: {share * 100:.1f}%")
        st.progress(float(share))

    st.divider()

    # Comparative Median Profile Table 
    st.subheader("Segment Profiles (Medians)")
    st.caption("Comparative behavioral metrics across standard customer personas")

    profiles = inliers.groupby("Cluster")[feature_cols].median().T
    profiles.columns = [CLUSTER_PERSONAS.get(col, f"Cluster {col}") for col in profiles.columns]

    # Make feature names human-readable (e.g., "AverageOrderValue" -> "Average Order Value")
    profiles.index = [re.sub(r'(?<!^)(?=[A-Z])', ' ', str(idx)) for idx in profiles.index]

    styled_profiles = profiles.style.highlight_max(
        axis=1, 
        props="background-color: #EFF6FF; color: #122969; font-weight: 600;"
    )

    for idx in profiles.index:
        # Currency formatting for spending metrics
        if any(keyword in idx for keyword in ["Monetary", "Value"]):
            styled_profiles = styled_profiles.format(formatter="${:,.2f}", subset=pd.IndexSlice[idx, :])
        
        # Percentage formatting for rates/ratios
        elif any(keyword in idx for keyword in ["Rate", "Ratio"]):
            styled_profiles = styled_profiles.format(formatter="{:.1%}", subset=pd.IndexSlice[idx, :])
        
        # Standard comma-separated formatting for counts (Frequency, Recency, Items)
        else:
            styled_profiles = styled_profiles.format(formatter="{:,.1f}", subset=pd.IndexSlice[idx, :])

    st.dataframe(styled_profiles, use_container_width=True)
    st.divider()

    # Single Segment Inspector
    st.subheader("Segment Deep-Dive")
    st.caption("Select a persona to investigate its defining behavioral characteristics.")

    clusters = sorted(inliers["Cluster"].unique())

    selected_cluster = st.selectbox(
        "Active Segment", 
        options=clusters,
        index=clusters.index(2) if 2 in clusters else 0,
        format_func=lambda x: (
        f"Cluster {x} — "
        f"{CLUSTER_PERSONAS.get(int(x), 'Unnamed Segment')}"
    )
    )
    
    cluster_data = inliers[inliers["Cluster"] == selected_cluster]
    
    col1, col2 = st.columns([1, 2.5], gap="large")

    # Cluster Card
    with col1:
        st.markdown(f"#### **Cluster {selected_cluster} Overview**")
        
        st.metric(
            "Segment Population", 
            f"{len(cluster_data):,}", 
            help="Total customers assigned to this cluster."
        )
        st.metric(
            "Population Share", 
            f"{(len(cluster_data) / len(results_df)) * 100:.1f}%", 
            help="Percentage of the total customer base."
        )
        
        profile_info = CLUSTER_PROFILES.get(
            int(selected_cluster),
            {
                "title": "Standard Cluster",
                "description": "No qualitative profile information is available for this segment."
            }
        )

        st.markdown(
            f"""
            <div class="cluster-profile-card">
                <span class="cluster-profile-title">Persona: {profile_info['title']}</span>
                <span class="cluster-profile-text">{profile_info['description']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Behavioral Table
    with col2:
        st.markdown("#### Behavioral Signature")
        
        cluster_medians = cluster_data[feature_cols].median()
        pop_medians = results_df[feature_cols].median()
        
        comp_df = pd.DataFrame({
            "Segment Median": cluster_medians,
            "Population Median": pop_medians
        })
        
        comp_df["Variance"] = (comp_df["Segment Median"] / comp_df["Population Median"].replace(0, np.nan)) - 1
                
        def style_variance(val):
            if pd.isna(val):
                return ""
            color = "#2563EB" if val > 0 else "#64748B"
            return f"color: {color}; font-weight: 600;"

        styled_comp = comp_df.style.map(style_variance, subset=["Variance"])
        
        # Format rows dynamically based on metric type
        for idx in comp_df.index:
            if any(keyword in idx for keyword in ["Monetary", "Value"]):
                styled_comp = styled_comp.format(formatter="${:,.2f}", subset=pd.IndexSlice[idx, ["Segment Median", "Population Median"]])
            elif any(keyword in idx for keyword in ["Rate", "Ratio"]):
                styled_comp = styled_comp.format(formatter="{:.1%}", subset=pd.IndexSlice[idx, ["Segment Median", "Population Median"]])
            else:
                styled_comp = styled_comp.format(formatter="{:,.1f}", subset=pd.IndexSlice[idx, ["Segment Median", "Population Median"]])
        
        # Format the variance column as a +/- percentage
        styled_comp = styled_comp.format(formatter="{:+.1%}", subset=["Variance"])
        
        st.dataframe(styled_comp, use_container_width=True)


def render_anomalies(results_df: pd.DataFrame, MultiMethod_Anomaly: pd.DataFrame, feature_cols: list):
    """Page 3: Isolation Forest Anomaly Detection."""
    render_header("Behavioral Anomalies")
    st.caption("Identify customers whose purchasing behavior is unusually different from the population.")

    # Metrics Part
    anomalies_df = results_df[results_df["Is_Anomaly"]].sort_values("Anomaly_Score", ascending=False)

    total_customers = len(results_df)
    flagged_count = len(anomalies_df)
    high_conf_count = MultiMethod_Anomaly['High_Confidence_Anomaly'].sum()
    max_score = results_df["Anomaly_Score"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flagged Outliers", f"{flagged_count:,}", help="Total Isolation Forest flags")
    c2.metric("High-Confidence Flags", f"{high_conf_count:,}", help="Multi-method consensus outliers", delta_color="inverse")
    c3.metric("Population Share", f"{(flagged_count / total_customers) * 100:.2f}%")
    c4.metric("Highest Score", f"{max_score:.4f}", help="Peak relative unusualness score")

    st.divider()

    # PCA Map with Anomalies
    st.subheader("PCA Anomaly Projection")

    pca_df = get_pca_coordinates(results_df, feature_cols)
    pca_df["Is_Anomaly"] = results_df["Is_Anomaly"].map({True: "Anomaly", False: "Normal"})

    fig_pca = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color="Is_Anomaly",
        color_discrete_map={"Normal": NEUTRAL_COLOR, "Anomaly": ANOMALY_COLOR},
        symbol="Is_Anomaly",
        symbol_map={"Normal": "circle", "Anomaly": "diamond"},
        opacity=0.8,
        template="plotly_white",
        title='2D PCA Behavioral Anomaly Landscape'
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # Anomaly Score Distribution
    st.subheader("Anomaly Score Distribution")

    fig = px.histogram(
        results_df,
        x="Anomaly_Score",
        color="Is_Anomaly",
        color_discrete_map={True: ANOMALY_COLOR, False: NEUTRAL_COLOR},
        barmode="overlay",
        template="plotly_white",
        labels={"Anomaly_Score": "Anomaly Score (Higher = More Unusual)"},
        title='Isolation Forest Anomaly Score Density'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Investigation Table
    st.subheader("Triage & Investigation Queue")
    st.caption("High-priority accounts flagged for manual review, sorted by algorithmic deviation severity.")

    # Join High_Confidence_Anomaly column onto anomalies_df using 'Customer ID' index
    triage_df = anomalies_df.join(
        MultiMethod_Anomaly[["High_Confidence_Anomaly"]], 
        how="left"
    )

    triage_df = triage_df.sort_values(by="Anomaly_Score", ascending=False).reset_index()

    # Filter display columns safely
    core_cols = ["Customer ID", "Cluster", "High_Confidence_Anomaly", "Anomaly_Score"]
    rfm_cols = [c for c in ["Monetary", "Frequency", "Recency"] if c in triage_df.columns]
    display_df = triage_df[core_cols + rfm_cols]

    # render grid with explicit CheckboxColumn configuration
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Customer ID": st.column_config.TextColumn(
                "Account ID",
                width="medium",
                help="Unique customer identifier."
            ),
            "Cluster": st.column_config.NumberColumn(
                "Assigned Persona",
                format="Cluster %d",
                width="small"
            ),
            "Anomaly_Score": st.column_config.NumberColumn(
                "Deviation Score",
                help="Relative unusualness (Higher = More extreme edge case).",
                format="%.4f"
            ),
            "High_Confidence_Anomaly": st.column_config.CheckboxColumn(
                "High Confidence",
                help="Multi-method consensus anomaly flag.",
                width="small"
            ),
            "Monetary": st.column_config.NumberColumn(
                "Lifetime Spend",
                format="$%,.2f",
            ),
            "Frequency": st.column_config.NumberColumn(
                "Order Count",
                format="%d"
            ),
            "Recency": st.column_config.NumberColumn(
                "Days Inactive",
                format="%d days"
            )
        }
    )


def render_customer_lookup(results_df: pd.DataFrame, feature_cols: list):
    """Page 4: Individual Customer Audit."""
    render_header("Customer Lookup")
    st.caption("Search individual customer accounts for detailed profile inspection.")

    customer_ids = results_df.index.tolist()
    selected_id = st.selectbox("Search Customer ID", customer_ids)

    if selected_id is not None:
        customer_row = results_df.loc[selected_id]

        # Metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"**Segment**\n\nCluster {int(customer_row['Cluster'])}")
        with c2:
            if customer_row["Is_Anomaly"]:
                st.error("**Anomaly Status**\n\n⚠ Flagged")
            else:
                st.success("**Anomaly Status**\n\nNormal")
        with c3:
            st.warning(f"**Anomaly Score**\n\n{customer_row.get('Anomaly_Score', 0.0):.4f}")

        st.divider()

        # Behavioral Profile part
        st.subheader("Behavioral Profile")
        profile_series = customer_row[feature_cols]
        st.dataframe(pd.DataFrame(profile_series).rename(columns={selected_id: "Observed Value"}), use_container_width=True)

        # Interpretation part
        st.subheader("Interpretation")
        if customer_row["Is_Anomaly"]:
            st.write(
                f"Customer **{selected_id}** belongs to **Cluster {int(customer_row['Cluster'])}** "
                f"and is flagged as behaviorally unusual relative to the population. "
                f"Review underlying features before taking automated action."
            )
        else:
            st.write(
                f"Customer **{selected_id}** is assigned to **Cluster {int(customer_row['Cluster'])}** "
                f"and exhibits standard purchasing patterns consistent with core segment behaviors."
            )
