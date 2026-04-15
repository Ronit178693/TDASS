# Core neural network architecture for the Oracle AI subsystem's predictive engine.
"""
# This module defines the LSTMPredictor, a recurrent neural network designed to capture temporal patterns in agent behavior.
lstm_predictor.py — B2: Pure LSTM Action Predictor (2-Layer)
===================================================
# It utilizes a stacked, bidirectional LSTM to process tactical histories and output discrete action probabilities.
Standard 2-layer LSTM architecture.
# The weights are often initialized from high-fidelity simulations to provide a "baseline" of expected enemy doctrine.
Matches high-fidelity trained weights.
"""

# Import the core PyTorch library for tensor calculations and gradient tracking.
import torch
# Import the neural network module to access standard layers like LSTM, Linear, and ReLU.
import torch.nn as nn

# Primary model class that inherits from PyTorch's base Module, allowing for easy training and serialization.
class LSTMPredictor(nn.Module):
    """
    Standard LSTM-based next-action predictor.
    This architecture is optimized for sequence classification where time-series dependency is critical.
    """

    # Constructor method where we define the architectural hyperparameters and initialize the layers.
    def __init__(
        self,
        # The number of input features per timestep (e.g., coordinates, HP, ammo).
        input_size: int,
        # The dimensionality of the hidden states; larger values allow for more complex pattern recognition.
        hidden_size: int = 128,
        # The number of stacked LSTM layers; 2 layers help capture both micro and macro temporal dependencies.
        num_layers: int = 2,
        # The number of target classes (e.g., 6 possible movement/attack actions).
        num_classes: int = 6,
        # Dropout rate used during training to prevent overfitting by randomly zeroing activations.
        dropout: float = 0.2,
    ):
        # Call the parent class constructor to register this class as a valid PyTorch network.
        super().__init__()
        # Store the hidden size for use in state initialization or debugging.
        self.hidden_size = hidden_size
        # Record the number of layers to determine if dropout can be safely applied within the LSTM block.
        self.num_layers = num_layers

        # Define the core Recurrent Neural Network component.
        # We use LSTM (Long Short-Term Memory) instead of Vanilla RNN to avoid the vanishing gradient problem.
        self.lstm = nn.LSTM(
            # Pass the count of tactical features expected in each frame.
            input_size=input_size,
            # Set the internal representation size for the LSTM's memory cells.
            hidden_size=hidden_size,
            # Stack multiple layers to build a hierarchy of temporal features.
            num_layers=num_layers,
            # Set batch_first=True so the input shape is (Batch, Sequence, Features), which is standard for tabular data.
            batch_first=True,
            # Enable Bidirectional processing to allow the model to look at sequences from both directions during training.
            bidirectional=True,
            # Apply dropout between recurrent layers (only if we have more than one layer).
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Define the 'Classification Head' which translates the LSTM's high-level features into final predictions.
        # We use a Sequential block to chain multiple operations together.
        self.classifier = nn.Sequential(
            # The first Linear layer bridges the gap between the hidden space and the output space.
            # We multiply hidden_size by 2 because the Bidirectional LSTM concatenates forward and backward outputs.
            nn.Linear(hidden_size * 2, hidden_size),
            # ReLU (Rectified Linear Unit) introduces non-linearity, allowing the model to learn non-trivial decision boundaries.
            nn.ReLU(),
            # Additional dropout layer for regularization before the final output mapping.
            nn.Dropout(dropout),
            # The final Linear layer projects the data down to the specific number of possible action classes (e.g., 0-5).
            nn.Linear(hidden_size, num_classes)
        )

    # The forward pass logic defines how data flows from the input through the layers to the final output.
    def forward(self, x):
        # Pass the input batch through the LSTM layers.
        # The underscore ignores the final hidden/cell states as we only need the sequence outputs.
        lstm_out, _ = self.lstm(x)
        
        # Take only the information from the VERY LAST timestep in the history window. 
        # This represents the cumulative "knowledge" of the sequence up to the current moment.
        last_step = lstm_out[:, -1, :]
        
        # Feed the summarized history into the classifier head to get class logits (unnormalized probabilities).
        # These logits will be used by CrossEntropyLoss during training or Softmax during inference.
        return self.classifier(last_step)
