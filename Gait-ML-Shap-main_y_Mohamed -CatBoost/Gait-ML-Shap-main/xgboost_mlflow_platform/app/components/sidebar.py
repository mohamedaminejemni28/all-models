from __future__ import annotations

import pandas as pd
import streamlit as st

from app.utils.dataframe import metric_range


def render_filters(runs: pd.DataFrame) -> dict:
    st.sidebar.header("Filters")

    if runs.empty:
        return {}

    datasets = sorted(runs["dataset"].dropna().unique().tolist())
    selected_datasets = st.sidebar.multiselect("Dataset", datasets, default=datasets)

    run_query = st.sidebar.text_input("Run name contains", "")

    feature_values = pd.to_numeric(runs["num_features"], errors="coerce").dropna()
    if feature_values.empty:
        feature_range = (0, 100)
    else:
        feature_range = (int(feature_values.min()), int(feature_values.max()))
    selected_feature_range = st.sidebar.slider(
        "Number of features",
        min_value=feature_range[0],
        max_value=feature_range[1],
        value=feature_range,
    )

    cv_range = st.sidebar.slider("CV accuracy", 0.0, 1.0, metric_range(runs, "cv_accuracy"), 0.01)
    test_range = st.sidebar.slider("Test accuracy", 0.0, 1.0, metric_range(runs, "test_accuracy"), 0.01)
    sensitivity_range = st.sidebar.slider("Sensitivity", 0.0, 1.0, metric_range(runs, "sensitivity"), 0.01)
    specificity_range = st.sidebar.slider("Specificity", 0.0, 1.0, metric_range(runs, "specificity"), 0.01)

    return {
        "datasets": selected_datasets,
        "run_query": run_query.strip().lower(),
        "feature_range": selected_feature_range,
        "cv_range": cv_range,
        "test_range": test_range,
        "sensitivity_range": sensitivity_range,
        "specificity_range": specificity_range,
    }


def apply_filters(runs: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if runs.empty or not filters:
        return runs

    filtered = runs.copy()
    if filters["datasets"]:
        filtered = filtered[filtered["dataset"].isin(filters["datasets"])]

    if filters["run_query"]:
        filtered = filtered[
            filtered["run_name"].astype(str).str.lower().str.contains(filters["run_query"], na=False)
        ]

    low_features, high_features = filters["feature_range"]
    filtered = filtered[
        pd.to_numeric(filtered["num_features"], errors="coerce").between(low_features, high_features)
    ]

    for column, filter_name in [
        ("cv_accuracy", "cv_range"),
        ("test_accuracy", "test_range"),
        ("sensitivity", "sensitivity_range"),
        ("specificity", "specificity_range"),
    ]:
        low, high = filters[filter_name]
        values = pd.to_numeric(filtered[column], errors="coerce")
        filtered = filtered[values.between(low, high)]

    return filtered

