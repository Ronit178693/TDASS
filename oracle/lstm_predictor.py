# Module docstring details for the predictor script.
"""
# The name of the script and its module tag.
lstm_predictor.py — B2: Pure LSTM Action Predictor (2-Layer)
# Separator for the description block.
===================================================
# States that this uses a basic two-layer LSTM.
Standard 2-layer LSTM architecture.
# States it's built to match pre-trained weights.
Matches high-fidelity trained weights.
# Close out the docstring block.
"""
# Empty visual separation line.

# Imports the base torch library.
import torch
# Imports the specifically required neural network module layers.
import torch.nn as nn
# Another empty line.

# Create the primary prediction class which extends nn.Module.
class LSTMPredictor(nn.Module):
# Definition block comments for the class.
    """
# Details that this architecture predicts the next-action of units.
    Standard LSTM-based next-action predictor (2-layer).
# Closes the internal block.
    """
# Starts a new line.

# Function definition for class creation and variable assignments.
    def __init__(
# Required self pointer.
        self,
# Argument for dimensions size of individual inputs.
        input_size: int,
# Determines internal hidden layer representations; defaults to 128.
        hidden_size: int = 128,
# Determines the amount of recurrent stacking layers; defaults to 2.
        num_layers: int = 2,
# Determines output layer discrete bucket limits; defaults to 6 (actions).
        num_classes: int = 6,
# Float determining network dropout chance regularization; defaults to 0.2.
        dropout: float = 0.2,
# Ends argument definitions for this function.
    ):
# Invokes PyTorch nn.Module setup initializations to register submodules.
        super().__init__()
# Stores the passed-in hidden size dimensions to the object parameter state.
        self.hidden_size = hidden_size
# Stores the passed-in number of repeating layers to the object parameter state.
        self.num_layers = num_layers
# Opens a visual separation gap line.

# In-line text comment indicating sequence analysis component logic creation.
        # Core LSTM
# Assigns nn.LSTM recurrent network block function output layer class.
        self.lstm = nn.LSTM(
# Configures network component processing capabilities input limitations.
            input_size=input_size,
# Configures internal routing and computation mapping sizing limits.
            hidden_size=hidden_size,
# Feeds the recurrent layers parameter directly to nn function assignment argument.
            num_layers=num_layers,
# Adjusts standard input configurations shape ordering parameter checks.
            batch_first=True,
# Enables passing context recursively backwards inside individual series samples.
            bidirectional=True,
# Prevents PyTorch dropout warning crash if multiple layers are absent by checking beforehand.
            dropout=dropout if num_layers > 1 else 0.0
# Ends network component definition block wrapping structure format.
        )
# Empty line

# Text indicator for creating dense network processing translation mapping out.
        # Classification head
# Initializes a sequential order of layer execution mappings.
        self.classifier = nn.Sequential(
# Begins dense fully connected mapping operation taking in doubled size mappings.
            nn.Linear(hidden_size * 2, hidden_size),
# Relu acts as processing trigger function converting negative signal bounds formatting.
            nn.ReLU(),
# Prevents specific mapping paths from becoming exclusively necessary representations.
            nn.Dropout(dropout),
# End mapping function collapsing space into single discrete possibilities predictions.
            nn.Linear(hidden_size, num_classes)
# Finalizes class operations initialization definition scope closing bracket.
        )
# Gap

# The PyTorch primary operational data routing and parsing handler method logic definition.
    def forward(self, x):
# Extracts temporal representations via recurrent function output mapping operations.
        lstm_out, _ = self.lstm(x)
# Truncates list of historical operation mappings to specifically pick last sequence moment.
        last_step = lstm_out[:, -1, :]
# Returns the final parsed sequence output interpretation map mapping classifications.
        return self.classifier(last_step)
