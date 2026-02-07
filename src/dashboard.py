"""
Data Quality Dashboard with AI-Powered Anomaly Explanations.

Auto-generates visualizations and uses LLM to explain detected anomalies.

Run with: python -m streamlit run dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Any

from src.anomaly_detector import (
    compute_daily_profiles,
    detect_all_anomalies,
    detect_date_column,
)
from src.llm_explainer import (
    explain_batch,
    get_provider_and_model,
    load_llm_config,
)
from src.data_loader import (
    UniversalDataLoader,
    get_file_info,
)


# Page configuration
st.set_page_config(
    page_title="Data Quality Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data_files():
    """Load all supported data files from data directory."""
    data_dir = Path("data")
    supported_extensions = UniversalDataLoader.SUPPORTED_FORMATS.keys()
    all_files = []

    for ext in supported_extensions:
        all_files.extend(data_dir.glob(f"*{ext}"))

    return sorted(all_files)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def detect_anomalies_from_file(file_path: Path):
    """Detect anomalies from a data file."""
    loader = UniversalDataLoader()

    try:
        df = loader.load(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path.name}: {e}")
        return [], None

    date_column = detect_date_column(df)
    if date_column is None:
        return [], None

    profiles = compute_daily_profiles(df, date_column)
    if len(profiles) < 5:
        st.info(f"Not enough data to detect anomalies in {file_path.name} (only {len(profiles)} days, need at least 5)")
        return [], None

    anomalies = detect_all_anomalies(profiles)

    # Add source file to each anomaly
    for anomaly in anomalies:
        anomaly["source_file"] = file_path.name

    return anomalies, date_column


@st.cache_data
def get_llm_explanations(anomalies: list[dict[str, Any]]):
    """Get LLM explanations for anomalies."""
    if not anomalies:
        return []

    try:
        results = explain_batch(anomalies)
        return results
    except Exception as e:
        st.error(f"Error getting LLM explanations: {e}")
        return [(anomaly, None) for anomaly in anomalies]


def render_metrics(anomalies: list[dict[str, Any]]):
    """Render top-level metrics."""
    st.subheader("📊 Overview")

    total = len(anomalies)
    high = sum(1 for a in anomalies if a.get("severity") == "high")
    medium = sum(1 for a in anomalies if a.get("severity") == "medium")
    low = sum(1 for a in anomalies if a.get("severity") == "low")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Anomalies", total)
    with col2:
        st.metric("🔴 High Severity", high, delta=None, delta_color="inverse")
    with col3:
        st.metric("🟡 Medium Severity", medium)
    with col4:
        st.metric("🟢 Low Severity", low)


def render_anomaly_timeline(anomalies: list[dict[str, Any]]):
    """Render anomaly timeline chart."""
    if not anomalies:
        return

    st.subheader("📈 Anomaly Timeline")

    # Prepare data for timeline
    df = pd.DataFrame(anomalies)
    df["date"] = pd.to_datetime(df["date"])

    # Get breakdown by date and severity
    severity_counts = df.groupby([df["date"].dt.date, "severity"]).size().unstack(fill_value=0)
    severity_counts.index = pd.to_datetime(severity_counts.index)

    # Create complete date range to show all days
    date_range = pd.date_range(start=severity_counts.index.min(), end=severity_counts.index.max(), freq="D")
    severity_counts = severity_counts.reindex(date_range, fill_value=0)

    # Calculate total anomalies per day
    severity_counts["total"] = severity_counts.sum(axis=1)

    # Reset index for plotting
    severity_counts = severity_counts.reset_index()
    severity_counts.columns.name = None
    severity_counts.rename(columns={"index": "date"}, inplace=True)

    # Ensure all severity columns exist
    for sev in ["high", "medium", "low"]:
        if sev not in severity_counts.columns:
            severity_counts[sev] = 0

    # Create line chart with single trend line
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=severity_counts["date"],
        y=severity_counts["total"],
        mode="lines+markers",
        name="Total Anomalies",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=6),
        customdata=severity_counts[["high", "medium", "low"]],
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b><br>" +
            "Total: %{y}<br>" +
            "🔴 High: %{customdata[0]}<br>" +
            "🟡 Medium: %{customdata[1]}<br>" +
            "🟢 Low: %{customdata[2]}<br>" +
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title="Anomalies Over Time (Daily)",
        xaxis_title="Date",
        yaxis_title="Number of Anomalies",
        hovermode="closest",
        xaxis=dict(
            tickformat="%Y-%m-%d",
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_anomaly_distribution(anomalies: list[dict[str, Any]]):
    """Render anomaly type distribution."""
    if not anomalies:
        return

    st.subheader("📊 Anomaly Type Distribution")

    df = pd.DataFrame(anomalies)
    type_counts = df["type"].value_counts().reset_index()
    type_counts.columns = ["type", "count"]

    # Pie chart
    fig = px.pie(
        type_counts,
        values="count",
        names="type",
        title="Anomaly Types",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_anomaly_details(anomalies_with_explanations: list[tuple[dict, dict]]):
    """Render detailed anomaly list with AI explanations."""
    if not anomalies_with_explanations:
        st.info("No anomalies to display")
        return

    st.subheader("🔍 Anomaly Details with AI Explanations")

    for idx, (anomaly, explanation) in enumerate(anomalies_with_explanations):
        severity = anomaly.get("severity", "medium")
        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")

        with st.expander(
            f"{severity_emoji} {anomaly['type']} - {anomaly.get('source_file', 'Unknown')} ({anomaly['date']})",
            expanded=(idx < 3)  # Expand first 3 by default
        ):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Anomaly Details**")
                st.write(f"**Date:** {anomaly['date']}")
                st.write(f"**Severity:** {severity.upper()}")
                st.write(f"**Z-Score:** {anomaly.get('z_score', 'N/A'):.2f}")

                st.markdown("---")
                st.write(f"**Message:** {anomaly['message']}")
                st.write(f"**Expected:** {anomaly['expected']}")
                st.write(f"**Actual:** {anomaly['actual']}")

            with col2:
                if explanation:
                    st.markdown("**🤖 AI Analysis**")

                    # Root cause with confidence badge
                    confidence = explanation.get("confidence", "low")
                    confidence_color = {
                        "high": "🟢",
                        "medium": "🟡",
                        "low": "🔴"
                    }.get(confidence, "⚪")

                    st.markdown(f"**Root Cause** {confidence_color} *({confidence} confidence)*")
                    st.info(explanation["root_cause"])

                    # Suggested actions
                    st.markdown("**💡 Suggested Actions**")
                    for action in explanation["suggested_actions"]:
                        st.markdown(f"- {action}")

                    # Business impact
                    st.markdown("**📊 Business Impact**")
                    st.warning(explanation["business_impact"])

                    # Additional context
                    if explanation.get("additional_context"):
                        st.markdown("**ℹ️ Additional Context**")
                        st.caption(explanation["additional_context"])
                else:
                    st.warning("No AI explanation available")


def main():
    """Main dashboard application."""
    st.title("📊 Data Quality Dashboard")
    st.markdown("AI-powered anomaly detection and explanation platform")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Load LLM config
    try:
        config = load_llm_config()
        provider = config.get("provider", "unknown")
        model = config.get("model", "unknown")
        use_mock = config.get("use_mock", False)

        st.sidebar.success(f" LLM: {provider}/{model}")
        if use_mock:
            st.sidebar.info("Using mock explanations")
    except Exception as e:
        st.sidebar.error(f"LLM Config Error: {e}")
        use_mock = True

    st.sidebar.markdown("---")

    # Load data files
    data_files = load_data_files()

    if not data_files:
        st.error("No data files found in data/ directory")
        st.info("Supported formats: CSV, Parquet, JSON, Excel, Feather, HDF5, Pickle")
        st.info("Run `python -m src.data_generator` to generate sample data")
        return

    st.sidebar.header("📁 Data Sources")

    # Show file format info
    file_formats = {}
    for f in data_files:
        info = get_file_info(f)
        file_formats[f.name] = f"{info.get('format', 'Unknown')} ({info['size_mb']:.2f} MB)"

    selected_files = st.sidebar.multiselect(
        "Select files to analyze",
        options=[f.name for f in data_files],
        default=[f.name for f in data_files[:3]] if len(data_files) >= 3 else [f.name for f in data_files],
        format_func=lambda x: f"{x} - {file_formats.get(x, 'unknown')}"
    )

    if not selected_files:
        st.warning("Please select at least one file to analyze")
        return

    # Load and detect anomalies
    all_anomalies = []

    with st.spinner("Detecting anomalies..."):
        for file_name in selected_files:
            file_path = Path("data") / file_name
            anomalies, date_col = detect_anomalies_from_file(file_path)
            all_anomalies.extend(anomalies)

    if not all_anomalies:
        st.info("No anomalies detected in the selected files")
        return

    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")

    # Severity filter
    severity_filter = st.sidebar.multiselect(
        "Severity",
        options=["high", "medium", "low"],
        default=["high", "medium", "low"],
    )

    # Type filter
    all_types = list(set(a["type"] for a in all_anomalies))
    type_filter = st.sidebar.multiselect(
        "Anomaly Type",
        options=sorted(all_types),
        default=sorted(all_types)[:10],  # Default to first 10 types
    )

    # Date range filter
    dates = [pd.to_datetime(a["date"]) for a in all_anomalies]
    min_date = min(dates).date()
    max_date = max(dates).date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Apply filters
    filtered_anomalies = [
        a for a in all_anomalies
        if a["severity"] in severity_filter
        and a["type"] in type_filter
        and (
            len(date_range) == 2
            and date_range[0] <= pd.to_datetime(a["date"]).date() <= date_range[1]
        )
    ]

    # Show filtered count
    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Anomalies", len(filtered_anomalies))

    # Render metrics
    render_metrics(filtered_anomalies)

    st.markdown("---")

    # Render visualizations
    col1, col2 = st.columns(2, gap="large")

    with col1:
        render_anomaly_timeline(filtered_anomalies)

    with col2:
        render_anomaly_distribution(filtered_anomalies)

    st.markdown("---")

    # Get LLM explanations
    explain_limit = st.sidebar.slider(
        "Max anomalies to explain",
        min_value=1,
        max_value=min(50, len(filtered_anomalies)),
        value=min(10, len(filtered_anomalies)),
    )

    with st.spinner(f"Generating AI explanations for top {explain_limit} anomalies..."):
        # Sort by severity and take top N
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_anomalies = sorted(
            filtered_anomalies,
            key=lambda x: (severity_order.get(x["severity"], 3), x["date"]),
            reverse=True,
        )[:explain_limit]

        anomalies_with_explanations = get_llm_explanations(sorted_anomalies)

    # Render anomaly details with explanations
    render_anomaly_details(anomalies_with_explanations)

    # Footer
    st.markdown("---")
    st.caption(f"Dashboard generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
