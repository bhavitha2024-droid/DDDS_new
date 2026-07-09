"""
model.py

The temporal core of this project, and the main architectural upgrade over the base
paper: a Long Short-Term Memory network that consumes a sequence of behavioral feature
vectors (EAR/MAR/head-pose over ~1 second) and predicts a drowsiness class, instead of
the base paper's per-frame CNN/SSD object-detection classification.
"""

import torch
import torch.nn as nn


class DrowsinessLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * direction_factor, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, sequence_length, input_dim)
        returns logits: (batch, num_classes)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last timestep's hidden representation (summarizes the whole window)
        last_out = lstm_out[:, -1, :]
        logits = self.classifier(last_out)
        return logits


if __name__ == "__main__":
    # quick shape sanity check
    model = DrowsinessLSTM()
    dummy = torch.randn(8, 30, 7)  # batch=8, seq_len=30, features=7
    out = model(dummy)
    print("Output shape:", out.shape)  # expected: (8, 3)
