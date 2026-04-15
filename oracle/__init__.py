# Initialization for the TDASS Oracle Intelligent Subsystem.
"""
# Focus: Predictive adversary behavior analysis and spatial intent forecasting.
oracle/ — Module 1: Intelligence & Enemy Prediction
=====================================================
# This package encapsulates the neural reasoning layers of the Tactical Decision Support System.
# It handles the full ML lifecycle from windowed sequence generation to real-time inference.

Core Sub-Modules:
# B1: Feature Engineering — Sliding window temporal sequencing.
  - data_loader.py:       Handles CSV parsing and feature normalization.
# B2: Behavioral Prediction — Recurrent neural architecture.
  - lstm_predictor.py:    Predicts discrete action sequences (UP, DOWN, FIRE).
# B3: Strategic Classification — Posture awareness.
  - intent_classifier.py: Classifies high-level intent (ATTACK, RETREAT).
# B4: Spatial Forecasting — Probability distribution maps.
  - heatmap_generator.py: Translates logits into 2D risk heatmaps.
# B5: MLOps — The training orchestration pipeline.
  - train_oracle.py:      Manages hardware-accelerated training and checkpointing.
# B6: Deployment — High-performance inference bridge.
  - inference_engine.py:  Provides a stateful, rolling-buffer interface for real-time applications.
"""
