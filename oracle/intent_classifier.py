"""
intent_classifier.py — B3: Pure LSTM Intent Classifier (2-Layer)
=====================================================
Standard 2-layer LSTM architecture for postural awareness.
Matches high-fidelity trained weights.
"""

import torch
import torch.nn as nn


class IntentClassifier(nn.Module):
    """
    Standard LSTM-based intent/posture classifier (2-layer).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        num_postures: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Feedforward encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU()
        )

        # Core LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_postures)
        )

    def forward(self, x):
        x = self.encoder(x)
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        return self.classifier(pooled)
