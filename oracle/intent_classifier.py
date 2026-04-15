# Specialized neural architecture for identifying high-level strategic "Postures" from unit behavior.
"""
# This module implements the IntentClassifier, which categorizes unit behavior into strategies like SCOUT, ATTACK, or RETREAT.
intent_classifier.py — B3: Pure LSTM Intent Classifier (2-Layer)
=====================================================
# It differs from the Action Predictor by using a mean-pooling strategy to capture the overall "vibe" of a trajectory.
Standard 2-layer LSTM architecture for postural awareness.
# Postural awareness is key for the Oracle to understand the 'Intent' behind a series of movements.
Matches high-fidelity trained weights.
"""

# Import the core PyTorch library for tensor computations and dynamic graph building.
import torch
# Import neural network modules for standard layers (Linear, LSTM, etc.).
import torch.nn as nn

# Model class designed for classification of overarching strategic intent.
class IntentClassifier(nn.Module):
    """
    Standard LSTM-based intent/posture classifier (2-layer).
    Unlike the action predictor which focuses on the transition to the 'next' step, 
    this model summarizes a whole sequence to determine the current 'posture'.
    """

    # Constructor used to define the layer types and their connectivity.
    def __init__(
        self,
        # The number of features expected at each timestep of the input window.
        input_size: int,
        # Internal dimensionality for the LSTM representation (smaller than the predictor to prevent overfitting on simple intents).
        hidden_size: int = 96,
        # Stacked layers for deep temporal feature extraction.
        num_layers: int = 2,
        # Total number of categories in the target space (e.g., SCOUT, ATTACK, etc.).
        num_postures: int = 5,
        # Regularization factor to improve generalization on unseen tactical data.
        dropout: float = 0.2,
    ):
        # Initialize the base module class.
        super().__init__()
        # Store hidden dimensions for future layer scaling or normalization and debugging.
        self.hidden_size = hidden_size

        # Pre-process the input sequence before it reaches the recurrent layer.
        # This acts as a 'Feature Extractor' that projects the raw data into a more useful embedding space.
        self.encoder = nn.Sequential(
            # Project the raw input features (e.g., 12 dims) into a latent 64-dimensional space.
            nn.Linear(input_size, 64),
            # ReLU creates the necessary non-linearity for the feature extractor.
            nn.ReLU()
        )

        # The core temporal engine.
        self.lstm = nn.LSTM(
            # Recurrently process the 64-D embeddings produced by our encoder.
            input_size=64,
            # Hidden size for the memory cells.
            hidden_size=hidden_size,
            # Depth of the LSTM stack.
            num_layers=num_layers,
            # Shapes expected as (Batch, Time, Features).
            batch_first=True,
            # Bidirectional LSTM allows the model to reconsider earlier steps in light of later ones.
            bidirectional=True,
            # Dropout specifically applied between recurrent layers for better convergence.
            dropout=dropout if num_layers > 1 else 0.0
        )

        # The mapping from temporal features to strategic classifications.
        self.classifier = nn.Sequential(
            # Combine the forward and backward components of the bidirectional LSTM.
            nn.Linear(hidden_size * 2, hidden_size),
            # Non-linear decision boundary.
            nn.ReLU(),
            # Dropout to prevent the model from memorizing specific training trajectories.
            nn.Dropout(dropout),
            # Map down to the specific number of posture classes.
            nn.Linear(hidden_size, num_postures)
        )

    # Logic for processing an input signal.
    def forward(self, x):
        # Step 1: Process each frame in the sequence through the feature encoder.
        x = self.encoder(x)
        # Step 2: Feed the encoded sequence into the LSTM. 
        # Here, lstm_out contains the hidden states for ALL timesteps.
        lstm_out, _ = self.lstm(x)
        
        # MEAN POOLING: Instead of taking just the last step, we average ALL steps in the window.
        # This is better for 'Intent' because a unit's strategy is reflected across the entire sequence, not just the final moment.
        pooled = torch.mean(lstm_out, dim=1)
        
        # Step 3: Classify the averaged sequence state into a posture probability map.
        return self.classifier(pooled)
