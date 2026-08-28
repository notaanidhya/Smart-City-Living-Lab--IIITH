"""
ml/models/mlp.py
================
Model A: Multi-Head MLP Issue Classifier

Architecture:
    Input (22) -> BN -> Dense(128) -> ReLU -> Dropout(0.3)
               -> Dense(64)  -> ReLU -> Dropout(0.2)
               -> Dense(32)  -> ReLU
               -> 6 independent Sigmoid heads (one per issue class)

Design rationale:
  - BatchNorm on input: the 22 raw features span wildly different scales
    (laplacian_variance in [0, 11000], fft_high_freq_ratio in [0, 0.84]).
    BN normalises them without requiring manual per-feature standardisation.
  - Modest depth (3 hidden layers): sufficient non-linearity for 22-dim input;
    deeper networks risk overfitting on 700 training samples.
  - Dropout: regularises the model; independently tuned per layer.
  - Independent sigmoid heads: supports multi-label output where multiple
    issues co-occur (e.g., blur + noise is common in our dataset).
"""

import torch
import torch.nn as nn

NUM_FEATURES = 22
NUM_CLASSES  = 6   # blur, underexposure, overexposure, noise, corruption, defect


class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim: int = NUM_FEATURES, hidden1: int = 128,
                 hidden2: int = 64, hidden3: int = 32, dropout1: float = 0.3,
                 dropout2: float = 0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
        )
        # One independent sigmoid output head per issue class
        self.heads = nn.ModuleList([nn.Linear(hidden3, 1) for _ in range(NUM_CLASSES)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 22)

        Returns
        -------
        torch.Tensor, shape (batch, 6)
            Sigmoid probabilities for each of the 6 issue classes.
        """
        x = self.input_bn(x)
        features = self.backbone(x)
        out = torch.cat([torch.sigmoid(head(features)) for head in self.heads], dim=1)
        return out   # shape: (batch, 6)
