"""Data loading and preprocessing for CNN gait experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


FIRST_FEATURE = "Pelv_Angle_Y_MAX_SW"
LAST_FEATURE = "Hip_Angle_Z_OHS"


def read_dataset_config(config, project_root):
    train_path = Path(project_root) / config["SFS"]["dataset_train_file"]
    test_path = Path(project_root) / config["SFS"]["dataset_test_file"]
    label_column = config["SFS"]["labels_column"]
    classes = config["SFS"].get("classes", {})
    return train_path, test_path, label_column, classes


def load_processed_sheets(train_path, test_path, label_column, use_existing_train_test=True):
    train_excel = pd.ExcelFile(train_path)
    test_excel = pd.ExcelFile(test_path) if use_existing_train_test and test_path.exists() else None
    datasets = {}

    for sheet_name in train_excel.sheet_names:
        train_df = pd.read_excel(train_path, sheet_name=sheet_name)
        test_df = None
        if test_excel is not None and sheet_name in test_excel.sheet_names:
            test_df = pd.read_excel(test_path, sheet_name=sheet_name)

        if label_column not in train_df.columns:
            raise ValueError(f"Column '{label_column}' not found in sheet '{sheet_name}'")

        datasets[sheet_name] = (train_df, test_df)

    return datasets


def feature_columns(df):
    if FIRST_FEATURE in df.columns and LAST_FEATURE in df.columns:
        start = df.columns.get_loc(FIRST_FEATURE)
        end = df.columns.get_loc(LAST_FEATURE)
        return list(df.columns[start:end + 1])
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def prepare_arrays(train_df, test_df, label_column, random_state, test_size):
    train_df = train_df.copy()
    train_df[label_column] = train_df[label_column].astype(str).str.strip()
    features = feature_columns(train_df)

    if test_df is None:
        train_part, test_part = train_test_split(
            train_df,
            test_size=test_size,
            random_state=random_state,
            stratify=train_df[label_column],
        )
    else:
        test_part = test_df.copy()
        test_part[label_column] = test_part[label_column].astype(str).str.strip()
        train_part = train_df

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_part[label_column])
    y_test = encoder.transform(test_part[label_column])

    X_train = train_part[features].apply(pd.to_numeric, errors="coerce")
    X_test = test_part[features].apply(pd.to_numeric, errors="coerce")
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_test = X_test.fillna(medians).fillna(0.0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train.astype(np.float32), y_test.astype(np.float32), features, encoder, scaler


def rank_features(X, y, feature_names, random_state):
    scores = mutual_info_classif(X, y, discrete_features=False, random_state=random_state)
    ranked = np.argsort(scores)[::-1]
    return ranked.tolist(), pd.DataFrame({
        "Feature": np.array(feature_names)[ranked],
        "Mutual_Info": scores[ranked],
        "Rank": np.arange(1, len(ranked) + 1),
    })

