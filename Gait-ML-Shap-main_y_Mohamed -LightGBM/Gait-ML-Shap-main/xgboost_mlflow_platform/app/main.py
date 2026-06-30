from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))

from app.components.charts import render_metric_bars, render_run_scatter
from app.components.sidebar import apply_filters, render_filters
from app.config import APP_TITLE, default_tracking_uri
from app.services.artifact_service import (
    find_artifacts,
    get_local_artifact_path,
    read_excel_artifact,
)
from app.services.mlflow_service import connect, load_artifact_index


st.set_page_config(page_title=APP_TITLE, page_icon="XGB", layout="wide")


@st.cache_data(show_spinner=False)
def cached_connect(tracking_uri: str):
    bundle = connect(tracking_uri)
    artifacts = load_artifact_index(bundle.client, bundle.runs)
    return bundle.experiments, bundle.runs, artifacts


def main() -> None:
    st.title(APP_TITLE)
    st.caption("Local dashboard for XGBoost runs, Top 3 Excel results, metrics, SHAP plots, and MLflow artifacts.")

    with st.sidebar:
        st.header("MLflow")
        tracking_uri = st.text_input("Tracking URI", value=default_tracking_uri())
        if st.button("Reload MLflow data", use_container_width=True):
            cached_connect.clear()

    try:
        experiments, runs, artifacts = cached_connect(tracking_uri)
    except Exception as exc:
        st.error(f"Could not connect to MLflow: {exc}")
        st.stop()

    if runs.empty:
        st.warning("No active XGBoost runs were found in the selected MLflow tracking store.")
        st.stop()

    filters = render_filters(runs)
    filtered_runs = apply_filters(runs, filters)

    render_overview(filtered_runs, artifacts)
    render_tabs(filtered_runs, artifacts)


def render_overview(runs: pd.DataFrame, artifacts: pd.DataFrame) -> None:
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("XGBoost runs", len(runs))
    col2.metric("Datasets", runs["dataset"].nunique())
    col3.metric("Best test accuracy", _format_best(runs, "test_accuracy"))
    col4.metric("SHAP images", _count_shap_images(artifacts, runs))

    display_columns = [
        "dataset",
        "run_name",
        "best_model_name",
        "num_features",
        "cv_accuracy",
        "test_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "mcc",
        "status",
        "run_id",
    ]
    st.dataframe(
        runs[[column for column in display_columns if column in runs.columns]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "cv_accuracy": st.column_config.ProgressColumn("CV accuracy", min_value=0, max_value=1),
            "test_accuracy": st.column_config.ProgressColumn("Test accuracy", min_value=0, max_value=1),
            "sensitivity": st.column_config.ProgressColumn("Sensitivity", min_value=0, max_value=1),
            "specificity": st.column_config.ProgressColumn("Specificity", min_value=0, max_value=1),
        },
    )


def render_tabs(runs: pd.DataFrame, artifacts: pd.DataFrame) -> None:
    metrics_tab, top3_tab, compare_tab, shap_tab, artifact_tab = st.tabs(
        ["Metrics", "Top 3 Models", "Compare Runs", "SHAP Plots", "Artifacts"]
    )

    with metrics_tab:
        st.subheader("Metric Tables and Charts")
        render_metric_bars(runs)
        render_run_scatter(runs)

    with top3_tab:
        st.subheader("Top 3 XGBoost Models by Dataset")
        render_top3_models(runs, artifacts)

    with compare_tab:
        st.subheader("Visual Run Comparison")
        render_comparison(runs)

    with shap_tab:
        st.subheader("SHAP Summary Plots")
        render_shap_plots(runs, artifacts)

    with artifact_tab:
        st.subheader("MLflow Artifacts and Excel Outputs")
        render_artifacts(runs, artifacts)


def render_top3_models(runs: pd.DataFrame, artifacts: pd.DataFrame) -> None:
    if artifacts.empty:
        st.info("No artifacts were found for the selected runs.")
        return

    for dataset in sorted(runs["dataset"].dropna().unique()):
        st.markdown(f"#### {dataset}")
        dataset_runs = runs[runs["dataset"] == dataset]
        shown = False

        for _, run in dataset_runs.iterrows():
            run_artifacts = artifacts[artifacts["run_id"] == run["run_id"]]
            top3_files = find_artifacts(
                run_artifacts.to_dict("records"),
                suffixes=(".xlsx", ".xls"),
                contains="scores_xgboost_top3",
            )
            if not top3_files:
                continue

            top3_path = get_local_artifact_path(run["artifact_uri"], top3_files[0]["path"], run["run_id"])
            if top3_path is None:
                continue

            try:
                top3_df = read_excel_artifact(top3_path).head(3)
            except Exception as exc:
                st.warning(f"Could not read {top3_files[0]['path']}: {exc}")
                continue

            st.caption(f"Run: {run['run_name']}")
            st.dataframe(top3_df, use_container_width=True, hide_index=True)
            shown = True
            break

        if not shown:
            fallback = dataset_runs.sort_values(
                ["cv_accuracy", "test_accuracy"],
                ascending=False,
                na_position="last",
            ).head(3)
            st.dataframe(fallback, use_container_width=True, hide_index=True)


def render_comparison(runs: pd.DataFrame) -> None:
    run_labels = {
        f"{row.dataset} | {row.run_name} | {row.run_id[:8]}": row.run_id
        for row in runs.itertuples()
    }
    selected = st.multiselect(
        "Choose runs to compare",
        options=list(run_labels.keys()),
        default=list(run_labels.keys())[: min(4, len(run_labels))],
    )

    selected_ids = [run_labels[label] for label in selected]
    comparison = runs[runs["run_id"].isin(selected_ids)]
    if comparison.empty:
        st.info("Select at least one run to compare.")
        return

    render_metric_bars(comparison)
    columns = [
        "dataset",
        "run_name",
        "best_model_name",
        "num_features",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "cv_accuracy",
        "test_accuracy",
        "sensitivity",
        "specificity",
        "features",
    ]
    st.dataframe(
        comparison[[column for column in columns if column in comparison.columns]],
        use_container_width=True,
        hide_index=True,
    )


def render_shap_plots(runs: pd.DataFrame, artifacts: pd.DataFrame) -> None:
    if artifacts.empty:
        st.info("No artifacts were found for the selected runs.")
        return

    run_options = {
        f"{row.dataset} | {row.run_name} | {row.run_id[:8]}": row.run_id
        for row in runs.itertuples()
    }
    selected_label = st.selectbox("Run", options=list(run_options.keys()))
    selected_run_id = run_options[selected_label]
    run = runs[runs["run_id"] == selected_run_id].iloc[0]

    run_artifacts = artifacts[artifacts["run_id"] == selected_run_id]
    shap_images = find_artifacts(
        run_artifacts.to_dict("records"),
        suffixes=(".png", ".jpg", ".jpeg"),
        contains="xgboost",
    )
    shap_images = [artifact for artifact in shap_images if "shap" in artifact["path"].lower()]

    if not shap_images:
        st.info("No XGBoost SHAP image artifacts were found for this run.")
        return

    columns = st.columns(3)
    for index, artifact in enumerate(shap_images):
        local_path = get_local_artifact_path(run["artifact_uri"], artifact["path"], selected_run_id)
        if local_path is None or not local_path.exists():
            continue
        with columns[index % 3]:
            st.image(str(local_path), caption=artifact["file_name"], use_container_width=True)


def render_artifacts(runs: pd.DataFrame, artifacts: pd.DataFrame) -> None:
    if artifacts.empty:
        st.info("No artifacts were found for the selected runs.")
        return

    visible_artifacts = artifacts[artifacts["run_id"].isin(runs["run_id"])]
    artifact_query = st.text_input("Artifact name contains", "")
    if artifact_query:
        visible_artifacts = visible_artifacts[
            visible_artifacts["path"].str.lower().str.contains(artifact_query.lower(), na=False)
        ]

    st.dataframe(
        visible_artifacts[["dataset", "run_name", "file_name", "path", "size", "run_id"]],
        use_container_width=True,
        hide_index=True,
    )

    excel_artifacts = visible_artifacts[
        visible_artifacts["path"].str.lower().str.endswith((".xlsx", ".xls"))
    ]
    if excel_artifacts.empty:
        return

    st.markdown("#### Excel Preview")
    labels = [
        f"{row.dataset} | {row.file_name} | {row.run_id[:8]}"
        for row in excel_artifacts.itertuples()
    ]
    selected = st.selectbox("Excel artifact", labels)
    selected_row = excel_artifacts.iloc[labels.index(selected)]
    local_path = get_local_artifact_path(
        selected_row["artifact_uri"],
        selected_row["path"],
        selected_row["run_id"],
    )
    if local_path is None:
        st.warning("Could not resolve this artifact to a local file.")
        return

    try:
        preview = pd.read_excel(local_path).head(100)
        st.dataframe(preview, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Excel artifact",
            data=Path(local_path).read_bytes(),
            file_name=Path(local_path).name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as exc:
        st.warning(f"Could not preview this Excel file: {exc}")


def _format_best(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return "-"
    value = pd.to_numeric(df[column], errors="coerce").max()
    if pd.isna(value):
        return "-"
    return f"{value:.1%}"


def _count_shap_images(artifacts: pd.DataFrame, runs: pd.DataFrame) -> int:
    if artifacts.empty:
        return 0
    visible = artifacts[artifacts["run_id"].isin(runs["run_id"])]
    paths = visible["path"].str.lower()
    return int((paths.str.contains("xgboost").fillna(False) & paths.str.contains("shap").fillna(False)).sum())


if __name__ == "__main__":
    main()
