"""Training, evaluation, and interpretation helpers for gait CNNs."""

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gaitml.deep_learning.cnn_models import TabularGaitCNN


def to_tensor_dataset(X, y):
    x_tensor = torch.tensor(X[:, None, :], dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(x_tensor, y_tensor)


def train_one_model(X_train, y_train, params, validation_size, random_state, device):
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )

    train_loader = DataLoader(
        to_tensor_dataset(X_train[train_idx], y_train[train_idx]),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        to_tensor_dataset(X_train[val_idx], y_train[val_idx]),
        batch_size=params["batch_size"],
        shuffle=False,
    )

    model = TabularGaitCNN(
        n_features=X_train.shape[1],
        hidden_channels=params["hidden_channels"],
        kernel_size=params["kernel_size"],
        dropout=params["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience_left = params["patience"]
    history = []

    for epoch in range(1, params["epochs"] + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, val_acc = evaluate_loss_accuracy(model, val_loader, criterion, device)
        history.append({"Epoch": epoch, "Train_Loss": float(np.mean(train_losses)), "Val_Loss": val_loss, "Val_Accuracy": val_acc})

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = params["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_loss


def evaluate_loss_accuracy(model, loader, criterion, device):
    model.eval()
    losses, y_true, y_pred = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            losses.append(criterion(logits, yb).item())
            pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            y_true.extend(yb.cpu().numpy().astype(int))
            y_pred.extend(pred)
    return float(np.mean(losses)), accuracy_score(y_true, y_pred)


def cross_val_accuracy(X, y, params, cv_splits, random_state, device):
    _, class_counts = np.unique(y, return_counts=True)
    cv_splits = int(min(cv_splits, class_counts.min()))
    if cv_splits < 2:
        return float("nan")

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_true_all, y_pred_all = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_params = dict(params)
        fold_params["epochs"] = max(10, min(params["epochs"], 40))
        model, _, _ = train_one_model(X[train_idx], y[train_idx], fold_params, 0.2, random_state + fold, device)
        preds = predict(model, X[val_idx], params["batch_size"], device)
        y_true_all.extend(y[val_idx].astype(int))
        y_pred_all.extend(preds.astype(int))
    return accuracy_score(y_true_all, y_pred_all)


def predict(model, X, batch_size, device):
    loader = DataLoader(to_tensor_dataset(X, np.zeros(len(X))), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            preds.extend((torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int))
    return np.array(preds)


def metric_row(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = (2 * ppv * sensitivity / (ppv + sensitivity)) if (ppv + sensitivity) else 0.0
    likelihood_ratio = ((1 - sensitivity) / specificity) if specificity else float("inf")
    return {
        "Test_Accuracy": accuracy_score(y_true, y_pred),
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "NPV": npv,
        "PPV": ppv,
        "Likelihood_Ratio": likelihood_ratio,
        "F1": f1,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Confusion_Matrix": str(confusion_matrix(y_true, y_pred, labels=[0, 1])),
    }


def permutation_importance(model, X, y, feature_names, batch_size, device, max_features=25, random_state=0):
    rng = np.random.default_rng(random_state)
    baseline = accuracy_score(y, predict(model, X, batch_size, device))
    rows = []
    for idx, feature in enumerate(feature_names[:max_features]):
        X_perm = X.copy()
        rng.shuffle(X_perm[:, idx])
        score = accuracy_score(y, predict(model, X_perm, batch_size, device))
        rows.append({"Feature": feature, "Importance": baseline - score, "Baseline_Accuracy": baseline, "Permuted_Accuracy": score})
    return pd.DataFrame(rows).sort_values("Importance", ascending=False)


def save_training_plot(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history["Epoch"], history["Train_Loss"], label="Train loss")
    plt.plot(history["Epoch"], history["Val_Loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
