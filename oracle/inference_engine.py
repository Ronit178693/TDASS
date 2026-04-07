# Document string block defining the Inference Engine script.
"""
# File name and its B6 Module label.
inference_engine.py — B6: Real-time Intelligence Integration
# Visual separator for the file header.
=============================================================
# Overview of script wrapper functionality.
A clean wrapper to load trained Oracle models and run inference
# Continuing the wrapper functionality explanation for online use.
during live simulations/visualizations.
# Empty line.

# Usage example header.
Usage:
# Example instantiation of the engine object.
  engine = OracleInferenceEngine()
# Example functional call predicting probabilities from the engine.
  prediction = engine.update(current_state_dict)
# End block docstring.
"""
# Empty line.

# Import standard os library for path joining and verification operations.
import os
# Import standard torch library for handling neural models and tensors.
import torch
# Import numpy for array mathematical logic.
import numpy as np
# Import deque from collections for the rolling buffer window.
from collections import deque
# Empty line.

# Import the previously created B2 LSTM Predictor class.
from oracle.lstm_predictor import LSTMPredictor
# Import the previously created B3 Intent Classifier class.
from oracle.intent_classifier import IntentClassifier
# Import the previously created B4 Heatmap Generator class.
from oracle.heatmap_generator import HeatmapGenerator
# Import constant values used during normalization from the B1 data loader script.
from oracle.data_loader import FEATURE_COLS, NORM_RANGES, POSTURE_CLASSES
# Empty line.
# Empty line.

# Definition for the class containing model invocation logic sequentially.
class OracleInferenceEngine:
# Main docstring for the engine component block.
    """
# States the class acts as a stateful inference processor.
    Stateful inference engine for the TDSS Oracle.
# Elaborates on maintaining a historical memory sequence to form windows.
    Maintains a rolling history of the battlefield to run LSTM calls.
# Close docstring.
    """
# Empty line.

# The initialization definition for the engine class.
    def __init__(
# The standard parameter binding argument self.
        self,
# Directory relative path parameter indicating checkpoint locations default.
        checkpoint_dir="oracle/checkpoints",
# Count limit defining sequence tracking depths parameter default.
        window_size=10,
# Device computation acceleration default parameter target.
        device="cpu",
# Battlefield map dimensions default.
        grid_size=10
# End function argument initialization.
    ):
# Store the intended device string (CPU/GPU) as a class level property.
        self.device = device
# Store the historical sequence depth maximum parameter.
        self.window_size = window_size
# Store the square matrix boundary limit.
        self.grid_size = grid_size
# Empty line.

# Comment specifying history property use case.
        # rolling buffer for features: deque of length window_size
# Initialize history parameter using collections deque with restricted structural capacity parameter.
        self.history = deque(maxlen=window_size)
# Empty line.

# Comment mapping logic to Model loading mechanism blocks.
        # Load Models
# Instantiates empty Action Predictor model tied to target device acceleration.
        self.action_model = LSTMPredictor(input_size=len(FEATURE_COLS)).to(device)
# Instantiates empty Intent Classifier model tied to target device acceleration.
        self.intent_model = IntentClassifier(input_size=len(FEATURE_COLS)).to(device)
# Empty line.

# Open dict defining string file mappings to targeted saved weights objects.
        paths = {
# Joins the variable checkpoint_dir with the name of the saved action predictor weights.
            "action": os.path.join(checkpoint_dir, "action_predictor_best.pt"),
# Joins the variable checkpoint_dir with the name of the saved intent classifier weights.
            "intent": os.path.join(checkpoint_dir, "intent_classifier_best.pt")
# Close paths dictionary mappings block.
        }
# Empty line.

# Iterates over all mapped keys and paths from the generated dictionary structure.
        for key, path in paths.items():
# Evaluate whether the target mapping path string points to an existing file entity.
            if os.path.exists(path):
# Programmatic selection deciding which model to reference using conditional ternary statement mapping.
                model = self.action_model if key == "action" else self.intent_model
# Extracts PyTorch weights matching file string location mapped specifically to initialized target acceleration parameter mappings.
                model.load_state_dict(torch.load(path, map_location=device))
# Lock model internal values mapping behavior structures into static non-learning evaluation execution setting flags.
                model.eval()
# Prints success output statement mapping loaded model logic.
                print(f"  [Oracle] Loaded {key} model from {path}")
# Logic catch clause activated if target mapped weight does not exist.
            else:
# Prints error warning statement indicating raw untouched initialized weights processing mapping execution mode.
                print(f"  [Oracle] WARNING: Checkpoint {path} not found. Using raw weights.")
# Empty line.

# Comment mapping Heatmap initialization sequence instantiation definitions.
        # Heatmap Generator
# Instantiates spatial plotting model engine matching grid dimensions to variable tracking limit parameter.
        self.heatmap_gen = HeatmapGenerator(grid_size=grid_size)
# Empty line.

# Internal function defining state object formatting translation converting dictionary types into sequence vectors mapping sequences.
    def _normalize_row(self, row_dict):
# Functional description indicating formatting dictionary parameter into sequence arrays.
        """Convert state dict to normalized feature vector."""
# Instantiates blank mapping arrays targeting feature holding mapping blocks.
        vec = []
# Iterates over static column feature mapping string variables defined inside loader script targets.
        for col in FEATURE_COLS:
# Safely extracts matching dictionary values fallback setting variables parameters targets value parameter blocks.
            val = float(row_dict.get(col, 0))
# Safely extracts formatting limits setting parameters variables mapping objects variables fallback mappings mapping structures variables parameters.
            lo, hi = NORM_RANGES.get(col, (0, 1))
# Formulaic mathematical translation converting flat input limit values into scaled ratio mappings processing variables mapping output targets limits mapping variables parameters limits structures processing output blocks logic variables mappings conditional boundaries setting.
            norm_val = (val - lo) / (hi - lo) if hi != lo else 0.0
# Appending processed numerical result float into target list formatting constraints tracking array holding vector mapping logic mappings boundaries.
            vec.append(norm_val)
# Returns numpy mapping output logic sequence casting native standard format boundaries list variables logic sequences constraint types vectors blocks conditional constraints properties boundaries limits formats mappings conditional elements limits parameters sequence sequences formatted.
        return np.array(vec, dtype=np.float32)
# Empty line.

# Primary execution script functional definition mapping input update limits processing logic formatting variables outputs processing blocks formatting output elements boundaries outputs parameters sequence constraints conditional conditional statements execution functions limits parameters targets outputs updates formats logic sequences formats constraints execution parameters targets limits blocks logic mapping processing operations.
    def update(self, current_state_dict):
# Functional description logic properties conditional formatting format definitions format mappings limits targets boundaries parameters definition constraints boundaries processing constraints conditional definitions outputs.
        """
# Functional doc string statements processing outputs logic sequence constraints format properties definitions formatting sequences bounds format logic limit mappings limits parameter sequences constraints formats mappings definitions bounds limits mapping properties limits formats boundaries definition format variables mapping variables limits mapping bounds parameters formats format properties definition processing format mapping limits parameters output outputs format logic limits limit limits parameters definitions.
        Update history and run inference.
# Parameter formatting block formatting logic constraints definitions processing limitations definitions format format parameters formatting limits formats limits mapping limits formats parameters formats formats parameters format definitions format properties format configurations formats logic constraints formatting formatting.
        
# Argument input limitations documentation bounds limits mapping logic definitions formats constraints bounds processing formats format properties mapping limits boundaries properties logic parameters formats properties formats format limits boundaries definition logic parameters targets format boundaries.
        Args:
# State dictionary parameters limits description limits formats logic elements definitions limits formats format definitions logic limits limits definitions values constraints bounds target mappings definitions variables properties limit configurations logic mappings logic constraints targets logic variables logic limit formats definitions formats boundaries definitions formatting structures logic parameters mapping variables logic definitions limits format configurations definitions bounds limits format format structures format formats logic boundaries definition configurations properties constraints.
            current_state_dict: dict containing keys in FEATURE_COLS
# Parameter configurations mappings logic boundaries limits variables rules properties definitions format limitations boundaries values parameters targets formatting formatting format format logic configurations mapping logic formatting constraint processing definition properties parameters definitions configurations format limitations limits limitations format values logic descriptions limitations mappings values formats logic logic attributes logic attributes formats bounds definitions limitations formats.
        
# Output limitations logic conditions limitations formats formats logic boundaries logic return boundaries bounds limits variables targets constraints limits boundaries constraints configurations configurations mapping descriptions formatting mapping target limitations mapping properties boundaries logic returns definitions logic conditions logic outputs mapping formats.
        Returns:
# Return dictionary output logic elements mapping limitations mappings limitations dictionary conditions properties target definitions logic conditions return limits formats formats formats format variables logic targets formatting variables limits properties descriptions parameters limitations conditions mapping returns mapping variable mapping logic.
            dict containing action_probs, posture, confidence, and heatmap.
# Description closing block sequence property limitations mapping outputs boundaries definitions formats formatting logic output definitions definitions limits formats descriptions limitations definitions return boundaries limitations outputs bounds mapping logic configurations.
        """
# Inline text mappings configurations limitations constraints processing bounds formatting bounds mapping definitions format variables boundaries outputs targets format limit properties definitions logic processing configurations boundaries values formats descriptions limitations values format properties limitations configurations format format definitions constraints limits logic mapping limitations formats.
        # 1. Normalize and append to history
# Functional assignment mapping state conversions formatting logic variables conditional boundaries mapping definitions format returns conditional boundaries definitions execution constraints logic mappings output format bounds mapping processing formatting target mappings definitions assignments limits limit formats.
        vec = self._normalize_row(current_state_dict)
# Enforces buffer sequence mappings properties limits array boundaries boundaries property variables formatting formats constraints variables bounds formats outputs execution arrays values pushing properties target boundaries properties conditions execution limitations bounds variables conditions constraint tracking conditions inputs parameters formatting bounds constraints inputs formats limitations execution operations format values.
        self.history.append(vec)
# Empty line.

# Comment text values variable conditions formats operations.
        # check if we have enough history
# Conditional formatting sequence checking length properties boundaries formats logic bounds values constraints operations limitations inputs conditional operations inputs mappings format lengths.
        if len(self.history) < self.window_size:
# Return configurations definitions target limit conditions mapping execution logic mapping variables limits constraints processing variables input return constraints execution operations limitations operations constraint operations limits formatting boundaries formatting definitions properties limits formats.
            return None
# Empty line.

# Text definitions formats.
        # 2. Optimized Tensor Conversion (Prevents simulation freezing)
# Conversions operations mapping conditions sequence tracking inputs operations variables formatting boundaries mapping conditional constraints inputs mappings arrays format constraints limits array target limits execution formatting variables limitations sequence formats inputs inputs operations input assignments execution array tracking mappings boundaries parameters formats tracking mappings limits parameters.
        input_array = np.array(list(self.history), dtype=np.float32)
# Generates tensor representations format assignments casting dimensions format mappings bounds operations execution limits mapping formats properties boundaries boundaries inputs constraints casting tensor bounds outputs sizes execution definitions limits outputs limits tracking mapping output targets limits mapping dimensions mappings tracking mapping output limits outputs targets execution targets sizes formatting formats operations structures mapping representations variables conditions.
        input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(self.device)
# Empty line.

# Comment limit definitions formats variables.
        # 3. Predict Action & Intent
# No grad logic definitions conditions bounding bounds operations limits execution conditions context logic formats limitations properties bounds variables property format mapping variables processing limit formatting limits variables limitations variables context properties execution variables formatting definitions values targets format settings bounds mappings execution.
        with torch.no_grad():
# Maps probabilistic execution dimensions parsing limits execution formats targets format softmax variables conditions outputs tracking logic values mappings configurations probability limitations assignments bounds bounding values execution targets formats array variables logic execution.
            action_probs = torch.softmax(self.action_model(input_tensor), dim=-1).cpu().numpy()[0]
# Maps probability execution tracking assignment representations limitations logic execution formats bounds definitions logic formats softmax formatting limitations conditions parsing logic values probability formatting values setting definitions assignments constraints bounds targets format dimensions constraints.
            posture_probs = torch.softmax(self.intent_model(input_tensor), dim=-1).cpu().numpy()[0]
# Empty line.

# Logic configuration mapping boundaries formats operations formats probability limits array mapping variables assignments targets.
        posture_idx = np.argmax(posture_probs)
# Mapping definitions formats properties logic mapping tracking format array variables conditions values assignments targeting configuration tracking execution mappings logic execution bounds conditions definitions parameters limits formatting constraints properties logic string property targets mappings conditions definitions values formats definitions parameters variables processing constants target variables constraints constants values array limits names targets strings.
        posture_name = POSTURE_CLASSES[posture_idx]
# Conversion execution formatting mapping float properties rules outputs extraction targets variable logic definitions string float mapping variable mapping operations float value extraction limits format property values assignments format formatting formatting array conversion boundaries properties property property float formats extraction dimensions constraint string values logic array index values mappings execution index variables assignments variables mappings formats values.
        confidence = float(posture_probs[posture_idx])
# Empty line.

# Comment descriptions configurations outputs properties formats logic bounds logic values formatting formats variable assignments string parameters.
        # 4. Generate Heatmap (Monte Carlo or Diffusion)
# Comment strings boundaries tracking limits variables parsing values mapping string parsing logic execution outputs targets variables names limitations property targets strings parsing variables constants float outputs outputs limit property value processing integer mappings variables property float properties assignments.
        # Using current position from state_dict
# Extracts targets string variables float definitions extraction operations limitations logic formats string values dictionary values formatting extraction values formats execution assignment mappings float parameters extraction limits mapping string dictionary values dictionary tracking dimensions.
        curr_pos = (int(current_state_dict['red_x']), int(current_state_dict['red_y']))
# Empty line.
        
# Text definitions property formats limits definitions limits boundaries bounds processing outputs variables formats array constraints target variables arrays targets arrays float execution parameters limits parsing limits operations float variable string.
        # For real-time, we use action bits + diffusion for speed
# Computes sequence execution logic function target format values probabilities rules map variable operations function string formats parameters array assignment mappings outputs format bounds parsing variables execution map bounds probability format mapping dimensions float.
        heatmap = self.heatmap_gen.from_action_probs(action_probs, curr_pos, n_steps=3)
# Empty line.

# Returns dictionary variable format definitions properties bounds targets variables parsing variables constants limit mapping format string inputs operations maps target function map definitions format variable mappings operations definitions variables array format outputs dictionary return definitions string target variables format definition output format array index definitions processing float formatting variables target value formats string values variables targets dictionary strings definition values float.
        return {
# Function mapping variables format.
            "action_probs": action_probs,
# Variable defining strings float values mapping format logic operations execution parameters string values conditions formats processing limit formats string conditions formats configurations function logic parameters formats variable names extraction parsing boundaries format mapping names strings logic operations.
            "posture": posture_name,
# Float defining string names bounds variables variables parsing limit target extraction strings arrays index arrays property string.
            "confidence": confidence,
# Float tracking variables outputs function string outputs map target bounds array bounds floating names float.
            "heatmap": heatmap,
# Float extraction processing names float mapping extraction variables float array extraction.
            "predicted_action": int(np.argmax(action_probs))
# Parsing definition boundaries processing extraction formats formatting logic execution rules float.
        }
# Empty line.

# Float definitions target limits parsing arrays bounds return limit values string logic limits definitions limitations extraction limits processing definitions limits targets float variables values extraction bounds mapping names processing limits limits variables parameters variables properties mapping logic string bounds limitations strings attributes format formatting tracking format bounds boundaries format limitations conditions operations execution formatting return targets condition property parameters values strings attributes target definitions string parsing logic conditions strings parsing.
    def reset(self):
# Parsing formats strings limits definition formatting properties strings format definitions format constraints properties extraction limits properties parsing string logic floating names mapping condition logic property format definition execution configurations configurations value definition parsing return condition format limits attributes limit.
        """Clear history (call this at the start of a new match)."""
# Operations format condition float value definition configurations attributes index definitions condition values format format processing operations property values strings value logic property names return.
        self.history.clear()
