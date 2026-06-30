from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


METRIC_COLUMNS = ["cv_accuracy", "test_accuracy", "sensitivity", "specificity", "f1", "mcc"]


def render_metric_bars(df: pd.DataFrame) -> None:
    available = [column for column in METRIC_COLUMNS if column in df.columns]
    if df.empty or not available:
        st.info("No metric values are available for the selected runs.")
        return

    chart_df = df[["run_name", "dataset", *available]].melt(
        id_vars=["run_name", "dataset"],
        var_name="metric",
        value_name="value",
    )
    chart_df = chart_df.dropna(subset=["value"])

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("run_name:N", title="Run", sort="-x"),
            color=alt.Color("metric:N", title="Metric"),
            row=alt.Row("dataset:N", title="Dataset"),
            tooltip=["dataset", "run_name", "metric", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=120)
    )
    st.altair_chart(chart, use_container_width=True)


def render_run_scatter(df: pd.DataFrame) -> None:
    required = {"cv_accuracy", "test_accuracy", "dataset", "run_name"}
    if df.empty or not required.issubset(df.columns):
        return

    chart_df = df.dropna(subset=["cv_accuracy", "test_accuracy"]).copy()
    if chart_df.empty:
        return

    chart = (
        alt.Chart(chart_df)
        .mark_circle(size=130)
        .encode(
            x=alt.X("cv_accuracy:Q", title="CV Accuracy", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("test_accuracy:Q", title="Test Accuracy", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("dataset:N", title="Dataset"),
            size=alt.Size("num_features:Q", title="# Features"),
            tooltip=[
                "dataset",
                "run_name",
                "best_model_name",
                "num_features",
                alt.Tooltip("cv_accuracy:Q", format=".3f"),
                alt.Tooltip("test_accuracy:Q", format=".3f"),
                alt.Tooltip("sensitivity:Q", format=".3f"),
                alt.Tooltip("specificity:Q", format=".3f"),
            ],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

