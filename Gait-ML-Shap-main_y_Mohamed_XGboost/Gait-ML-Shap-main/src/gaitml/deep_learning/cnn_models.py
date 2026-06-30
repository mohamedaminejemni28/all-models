"""CNN models for tabular gait feature tensors."""

import torch
from torch import nn


class TabularGaitCNN(nn.Module):
    """A compact 1D CNN for ordered biomechanical feature vectors."""

    def __init__(self, n_features, hidden_channels=32, kernel_size=3, dropout=0.3):
        super().__init__()
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)

