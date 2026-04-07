# Docstring documenting the intent classifier.
"""
# Name and module number of this file.
intent_classifier.py — B3: Pure LSTM Intent Classifier (2-Layer)
# Decorative separator for the module description.
=====================================================
# Brief overview of the LSTM's purpose and depth.
Standard 2-layer LSTM architecture for postural awareness.
# Explanation of the specific weights used.
Matches high-fidelity trained weights.
# End of the module docstring.
"""
# Blank line for formatting purposes.

# Import the PyTorch library to access neural network components.
import torch
# Import PyTorch's neural network module specifically.
import torch.nn as nn
# Another blank line for separation.

# Define the IntentClassifier class inheriting from nn.Module.
class IntentClassifier(nn.Module):
# Opening docstring for the class itself.
    """
# Description of the class's architecture.
    Standard LSTM-based intent/posture classifier (2-layer).
# Closing the docstring.
    """
# Blank line.

# Define the initialization function for the class.
    def __init__(
# The self parameter referencing the instance.
        self,
# The input_size parameter defining the input dimension.
        input_size: int,
# The hidden_size parameter, defaulting to 96.
        hidden_size: int = 96,
# The number of LSTM layers, defaulting to 2.
        num_layers: int = 2,
# The number of output classes (postures), defaulting to 5.
        num_postures: int = 5,
# The dropout rate to prevent overfitting, defaulting to 0.2.
        dropout: float = 0.2,
# Close the arguments for the init method.
    ):
# Call the parent class (nn.Module) initialization method.
        super().__init__()
# Save the hidden_size parameter to the instance.
        self.hidden_size = hidden_size
# Blank line.

# Comment identifying the encoder block.
        # Feedforward encoder
# Initialize the sequential encoder block.
        self.encoder = nn.Sequential(
# A linear transformation layer mapping input_size to 64.
            nn.Linear(input_size, 64),
# Output Activation via ReLU function.
            nn.ReLU()
# Close the sequence definition.
        )
# Blank line.

# Comment identifying the core LSTM.
        # Core LSTM
# Initialize the LSTM module.
        self.lstm = nn.LSTM(
# The input features to the LSTM (64, matching the encoder's output).
            input_size=64,
# The feature dimensionality of the hidden state.
            hidden_size=hidden_size,
# Provide the number of stacked LSTM layers.
            num_layers=num_layers,
# Indicate that the batch dimension is the first dimension.
            batch_first=True,
# Enable the bidirectional features of the LSTM.
            bidirectional=True,
# Set the dropout only if there's more than one layer.
            dropout=dropout if num_layers > 1 else 0.0
# Close the LSTM definition.
        )
# Blank line.

# Comment mapping to the classification head.
        # Classification head
# Initialize the fully connected classifier block.
        self.classifier = nn.Sequential(
# Linear layer mapping bidirectional output (hidden*2) to hidden size.
            nn.Linear(hidden_size * 2, hidden_size),
# Activation using ReLU.
            nn.ReLU(),
# Applying dropout for regularization.
            nn.Dropout(dropout),
# Final linear layer reducing hidden space to output postures number.
            nn.Linear(hidden_size, num_postures)
# Close classification block definition.
        )
# Blank line.

# Define the forward pass method.
    def forward(self, x):
# Pass the input x through the initial encoder block.
        x = self.encoder(x)
# Pass the encoded x into the LSTM block. Returns output and (h, c).
        lstm_out, _ = self.lstm(x)
# Pool the sequence outputs by taking the mean across the sequence dimension (1).
        pooled = torch.mean(lstm_out, dim=1)
# Pass the pooled result through the final classifier and return.
        return self.classifier(pooled)
