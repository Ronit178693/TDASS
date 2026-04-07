# File docstring for the final visualizer entry point.
"""
# Header for the visualizer file.
visualize_oracle.py — Cyber-Directive Edition (v17.0 RESTORED)
# Visual separator.
============================================================
# High-level aesthetic summary.
The absolute peak of visual tactical discovery. 
# Feature highlights.
Neon Glows | Animated Radar | Digital Filter | Live Health Overlays
# HUD feature highlights.
Full Intelligence HUD | Intent Dashboard | Battle Log (Removed)
# End block.
"""
# Empty line.

# Import main pygame library for 2D graphics and input.
import pygame
# Import system utilities for exit and paths.
import sys
# Import copy for state management.
import copy
# Import time for frame pacing and match delays.
import time
# Import json for possible data logging.
import json
# Import random for flicker effects.
import random
# Import numpy for math operations on the grid.
import numpy as np
# Empty line.

# ──────────────────────────────────────────────────────────────────────────
# Visual section for UI geometry configuration.
# 🛠️ CONFIGURATION & HUD DIMENSIONS
# ──────────────────────────────────────────────────────────────────────────
# Total horizontal pixel count.
SCREEN_WIDTH  = 1240
# Total vertical pixel count.
SCREEN_HEIGHT = 800
# Dimension of the battlefield grid.
GRID_SIZE     = 600
# Dimension of a single map square.
TILE_SIZE     = 60
# X-coordinate start for the side intelligence panel.
HUD_X         = 680
# Width of the side intelligence panel.
HUD_WIDTH     = 540
# Empty line.

# ──────────────────────────────────────────────────────────────────────────
# Color palette definition for the Cyber-Directive theme.
# --- THEME: NEON TACTICAL ---
# Primary dark backdrop color.
COLOR_BG      = (5, 8, 12)
# Muted grid line color.
COLOR_GRID    = (20, 35, 55)
# Vibrant blue for friendly identifiers.
COLOR_BLUE    = (0, 200, 255)
# High-intensity red for hostile identifiers.
COLOR_RED     = (255, 30, 80)
# Caution gold for resource crates.
COLOR_SUPPLY  = (255, 215, 0)
# Tech-green for UI accents and intel labels.
COLOR_INTEL   = (0, 255, 180)
# Off-white for readability.
COLOR_TEXT    = (200, 220, 240)
# Empty line.

# Main class for the application window and game loop.
class UltimateCommander:
# Constructor to set up pygame and simulation engine.
    def __init__(self):
# Initialize pygame services.
        pygame.init()
# Create the window surface.
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# Set the OS title bar.
        pygame.display.set_caption("☢️ TDSS CYBER-DIRECTIVE — RESTORED v17.0")
# Empty line.
        
# Initialize various typography styles.
        # --- FONTS ---
# Primary header font (Verdana).
        self.header_font = pygame.font.SysFont("Verdana", 26, bold=True)
# Sub-header font.
        self.sub_font    = pygame.font.SysFont("Verdana", 18, bold=True)
# Standard text font.
        self.font        = pygame.font.SysFont("Verdana", 14)
# Tech/Code style font for data readouts.
        self.small_font  = pygame.font.SysFont("Consolas", 12)
# High-density label font.
        self.label_font  = pygame.font.SysFont("Consolas", 11, bold=True)
# Empty line.

# Initialize the tactical simulation cores.
        # --- SIMULATION ---
# Instantiate the battlefield grid (no direct render to terminal).
        self.env = BattleEnv(render_mode=None)
# Instantiate the AI brain and Oracle engine.
        self.brain = TacticalBrain()
# Set initial state.
        self.env.reset()
# Empty line.
        
# Setup app state variables.
        # --- STATE ---
# Match counter.
        self.match_count = 1
# Track mouse cursor on grid.
        self.hover_pos = (0, 0)
# Main frame counter for animations.
        self.tick_count = 0
# Logic for moving digital overlays.
        self.scanline_offset = 0
# Empty line.
        
# Setup static visual overlays.
        # --- HUD SURFACES ---
# Create transparent CRT overlay.
        self.scanline_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
# Generate horizontal scanlines.
        for y in range(0, SCREEN_HEIGHT, 4):
# Draw semi-transparent black lines across the screen.
            pygame.draw.line(self.scanline_surf, (0, 0, 0, 45), (0, y), (SCREEN_WIDTH, y))
# Empty line.

# Internal method to draw the map tiles and AI heatmaps.
    # ────────────────────────────────────────────────────────────────────────
    # 🎨 RENDERING: BATTLEFIELD & NEON EFFECTS
    # ────────────────────────────────────────────────────────────────────────
    def _draw_grid(self, prediction):
# Docstring.
        """Draw tactical 10x10 grid with animated Oracle heatmap."""
# Calculate sinusoidal pulse for heatmap transparency.
        pulse_alpha = abs(int(100 * np.sin(self.tick_count * 0.1))) + 50
# Empty line.
        
# Render the mouse hover highlights.
        # Crosshair Highlight
        hr, hc = self.hover_pos
# Draw vertical row shadow.
        pygame.draw.rect(self.screen, (10, 25, 45), (40, 100 + hr*60, 600, 60))
# Draw horizontal column shadow.
        pygame.draw.rect(self.screen, (10, 25, 45), (40 + hc*60, 100, 60, 600))
# Empty line.

# Loop through 10x10 coordinate grid.
        for r in range(10):
            for c in range(10):
# Pull terrain type from environment data.
                t_type = self.env.terrain_map[r, c]
# Default color for plains.
                color = (15, 20, 30) # Plains
# Wall color.
                if t_type == 1: color = (50, 55, 70) # Wall
# Forest color.
                if t_type == 2: color = (20, 45, 20) # Forest
# Water color.
                if t_type == 3: color = (10, 30, 80) # Water
# Road color.
                if t_type == 4: color = (70, 70, 80) # Road
# Empty line.
                
# Calculate pixel rectangle.
                rect = (40 + c*60, 100 + r*60, 60, 60)
# Draw the colored tile.
                pygame.draw.rect(self.screen, color, rect)
# Draw the glowing grid border.
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 1)
# Empty line.

# Overlay predictive AI intel on the grid.
                # Heatmap Radar Sweep
                if prediction:
# Get probability score for this square.
                    prob = prediction['heatmap'][r, c]
# If risk is significant, draw the highlight.
                    if prob > 0.08:
# Calculate pulse and probability transparency.
                        alpha = int(min(255, prob * pulse_alpha * 2.5))
# Create temporary surface for transparency.
                        overlay = pygame.Surface((58, 58), pygame.SRCALPHA)
# Fill with golden risk color.
                        overlay.fill((255, 215, 0, alpha // 4))
# Blit the risk flare.
                        self.screen.blit(overlay, (41 + c*60, 101 + r*60))
# Logic for the "Unknown Intent" indicator.
                        # Animated '?'
                        if self.tick_count % 30 < 15:
# Render character.
                            q_txt = self.sub_font.render("?", True, (255, 255, 100))
# Draw at tile location.
                            self.screen.blit(q_txt, (60 + c*60, 115 + r*60))
# Empty line.

# Render the Fog of War shadow layer.
        # Fog of War
# Setup dark shadow tile.
        fog_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
# Dark navy with high opacity.
        fog_surf.fill((0, 0, 5, 210))
# Scan grid.
        for r in range(10):
            for c in range(10):
# If grid tile is NOT visible to blue team.
                  if not self.env.fog_map[r, c]:
# Blit shadow.
                      self.screen.blit(fog_surf, (40 + c*60, 100 + r*60))
# Empty line.

# Render the physical supply crates.
        # Active Supply Drops
        for drop in self.env.supply_drops:
# Extract coords.
            dr, dc = drop['pos']
# Draw white border crate.
            pygame.draw.rect(self.screen, (255, 255, 255), (40 + dc*60+15, 100 + dr*60+15, 30, 30))
# Draw golden inner crate.
            pygame.draw.rect(self.screen, COLOR_SUPPLY, (40 + dc*60+18, 100 + dr*60+18, 24, 24))
# Empty line.

# Internal method to draw units and combat VFX.
    def _draw_units(self):
# Docstring.
        """Render units with on-map Vitality bars and Neon Lasers."""
# Loop through all units on the field.
        for u in self.env.blue_units + self.env.red_units:
# Skip dead units.
            if not u['alive']: continue
# Get current position.
            r, c = u['pos']
# Identify team affiliation.
            is_blue = (u['team'] == 'blue')
# Empty line.
            
# Check if unit should be visible (Fog of War check).
            if is_blue or self.env.fog_map[r, c]:
# Calculate screen center point for the tile.
                center = (40 + c*60 + 30, 100 + r*60 + 30)
# Select unit glow color.
                main_color = COLOR_BLUE if is_blue else COLOR_RED
# Empty line.
                
# Multi-ring glowing effect for a "hologram" look.
                # Glowing Unit Icon
                for i in range(3):
# Draw outer circles.
                    pygame.draw.circle(self.screen, main_color, center, 24-i*2, 1)
# Draw core circle.
                pygame.draw.circle(self.screen, main_color, center, 18)
# Empty line.
                
# Overlay health visual directly on the grid.
                # Floating HP Bar
# Draw backdrop shadow.
                pygame.draw.rect(self.screen, (0, 0, 0), (40+c*60+5, 100+r*60+52, 50, 4))
# Draw green health segment (Width scaled to HP).
                pygame.draw.rect(self.screen, (0, 255, 100), (40+c*60+5, 100+r*60+52, int(0.5 * u['hp']), 4))
# Empty line.

# Combat line logic.
                # Combat VFX: Neon Flicker Laser
# Check if unit fired in the previous turn.
                if u.get('prev_action') == 5:
# Find potential targets.
                    opponents = self.env.red_units if is_blue else self.env.blue_units
# Filter for survivors.
                    living = [o for o in opponents if o['alive']]
# Target identification logic.
                    if living:
# Choose closest enemy for visual laser.
                        tgt = min(living, key=lambda o: abs(r-o['pos'][0]) + abs(c-o['pos'][1]))
# Get target coords.
                        tr, tc = tgt['pos']
# Target center point.
                        tgt_center = (40 + tc*60 + 30, 100 + tr*60 + 30)
# Pulsing flicker effect.
                        if self.tick_count % 4 < 2:
# Draw broad yellow glow laser.
                            pygame.draw.line(self.screen, (255, 255, 150), center, tgt_center, 6)
# Draw sharp white core beam.
                            pygame.draw.line(self.screen, (255, 255, 255), center, tgt_center, 2)
# Empty line.

# Internal method to draw the side intelligence center.
    # ────────────────────────────────────────────────────────────────────────
    # 🖥️ HUD: TACTICAL INTELLIGENCE DASHBOARD
    # ────────────────────────────────────────────────────────────────────────
    def _draw_hud(self, prediction):
# Docstring.
        """Render the high-density Oracle Intel panels."""
# Master panel geometry.
        # Panel Background
# Solid dark-gray backing.
        pygame.draw.rect(self.screen, (10, 15, 25), (HUD_X, 20, HUD_WIDTH, 760), border_radius=10)
# Glowing blue border.
        pygame.draw.rect(self.screen, (30, 50, 80), (HUD_X, 20, HUD_WIDTH, 760), 2, border_radius=10)
# Empty line.

# Intel Section 1: Oracle Predictions.
        # 1. ORACLE PREDICTIVE INTENT
        y_off = 40
# Header text.
        self.screen.blit(self.header_font.render("ORACLE INTEL DASHBOARD", True, COLOR_INTEL), (HUD_X + 25, y_off))
# Empty line.
        
# Extract current prediction strings and values.
        posture = prediction['posture'] if prediction else "INITIALIZING..."
        conf    = prediction['confidence'] if prediction else 0.0
# Empty line.
        
# Visual card for the primary intent result.
        # Large Intent Card
# Color code by aggression level.
        card_color = (255, 200, 0) if posture == "ATTACK" else (0, 200, 255)
# Draw dark inset.
        pygame.draw.rect(self.screen, (20, 30, 50), (HUD_X + 25, y_off + 50, 490, 100), border_radius=8)
# Label text.
        self.screen.blit(self.sub_font.render("PREDICTED ENEMY INTENT:", True, (150, 170, 190)), (HUD_X + 45, y_off + 65))
# Prediction result bracketed text.
        self.screen.blit(self.header_font.render(f"[ {posture} ]", True, card_color), (HUD_X + 45, y_off + 95))
# Empty line.
        
# Reliability visualization.
        # Confidence Bar
# Text label.
        self.screen.blit(self.small_font.render(f"ORACLE CONFIDENCE: {conf*100:.1f}%", True, COLOR_TEXT), (HUD_X + 25, y_off + 165))
# Black empty bar.
        pygame.draw.rect(self.screen, (0, 0, 0), (HUD_X + 25, y_off + 185, 490, 12))
# Colored filled bar segment.
        pygame.draw.rect(self.screen, card_color, (HUD_X + 25, y_off + 185, int(490 * conf), 12))
# Empty line.

# Intel Section 2: Hostile Status Tracking.
        # 2. ENEMY THREAT SCANNER
        y_off = 260
# Red alert header.
        self.screen.blit(self.sub_font.render("🔴 ENEMY THREAT STATUS", True, COLOR_RED), (HUD_X + 25, y_off))
# Empty line.
        
# Iterate through enemy list.
        for i, u in enumerate(self.env.red_units):
# Row geometry calculation.
            row_y = y_off + 40 + (i * 70)
# Dark background row.
            pygame.draw.rect(self.screen, (15, 20, 35), (HUD_X+25, row_y, 490, 60), border_radius=5)
# Empty line.
            
# Check if current unit is revealed to blue team sensors.
            r, c = u['pos']
            visible = self.env.fog_map[r, c]
# Empty line.
            
# Dead unit logic.
            if not u['alive']:
# Neutralization label.
                txt = "--- TARGET NEUTRALIZED ---"
                self.screen.blit(self.font.render(txt, True, (0, 255, 100)), (HUD_X + 150, row_y + 20))
# Fogged unit logic.
            elif not visible:
# Cloak warning.
                txt = "--- TARGET CLOAKED (IN FOG) ---"
                self.screen.blit(self.font.render(txt, True, (100, 110, 130)), (HUD_X + 140, row_y + 20))
# Revealed unit logic.
            else:
# Display coordinate data.
                self.screen.blit(self.small_font.render(f"ID: {u['id']} | POS: [{c},{r}]", True, COLOR_TEXT), (HUD_X + 40, row_y + 12))
# Display HP bar.
                # Mini HP Bar
                pygame.draw.rect(self.screen, (0, 0, 0), (HUD_X + 40, row_y + 35, 200, 8))
                pygame.draw.rect(self.screen, COLOR_RED, (HUD_X + 40, row_y + 35, int(2 * u['hp']), 8))
# Display Ammo count.
                self.screen.blit(self.small_font.render(f"AMMO: {u['ammo']}", True, (255, 150, 0)), (HUD_X + 260, row_y + 32))
# Empty line.

# Intel Section 3: Ground Sensor Readings.
        # 3. BATTLEFIELD SENSOR PROBE (Bottom Panel)
        y_off = 620
# Intel green header.
        self.screen.blit(self.sub_font.render("🔎 ORACLE RISK SIGNATURE", True, COLOR_INTEL), (HUD_X + 25, y_off))
# Hover position tracking.
        hr, hc = self.hover_pos
# Terrain data lookup.
        t_type = self.env.terrain_map[hr, hc]
        t_names = ["PLAINS", "WALL", "FOREST", "WATER", "ROAD"]
# Heatmap risk interpolation.
        risk = prediction['heatmap'][hr, hc] if prediction else 0
# Empty line.
        
# Draw inset panel.
        pygame.draw.rect(self.screen, (5, 30, 45), (HUD_X+25, y_off + 40, 490, 80), border_radius=5)
# Print coordinate readout.
        self.screen.blit(self.font.render(f"PROBE COORDINATES: [{hc}, {hr}]", True, COLOR_TEXT), (HUD_X + 45, y_off + 50))
# Print terrain classification.
        self.screen.blit(self.font.render(f"TERRAIN: {t_names[t_type] if t_type < 5 else 'UNKNOWN'}", True, COLOR_TEXT), (HUD_X + 45, y_off + 70))
# Print probability score.
        self.screen.blit(self.font.render(f"LOCAL THREAT PROBABILITY: {risk*100:.2f}%", True, (255, 255, 0)), (HUD_X + 45, y_off + 90))
# Empty line.

# Master execution loop.
    def run(self):
# Initialize frame rate anchor.
        clock = pygame.time.Clock()
# Loop control flag.
        running = True
# Empty line.
        
# Begin frame loop.
        while running:
# Tick counters for animation pacing.
            self.tick_count += 1
# Event polling loop.
            for event in pygame.event.get():
# Exit command check.
                if event.type == pygame.QUIT: running = False
# Mouse movement tracking for grid probe.
                if event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
# Check if mouse is within map bounds (40-640 x-range, 100-700 y-range).
                    if 40 < mx < 640 and 100 < my < 700:
# Convert pixels to grid coordinates.
                        self.hover_pos = ((my-100)//60, (mx-40)//60)
# Empty line.

# High-level AI inference processing.
            # --- BRAIN: TACTICAL ORACLE ---
# Identify primary red unit.
            red_u = next((u for u in self.env.red_units if u['alive']), None)
# Identify primary blue unit.
            blue_u = next((u for u in self.env.blue_units if u['alive']), None)
# Clear results.
            prediction = None
# Perform calculation only if units exist.
            if red_u and blue_u:
# Prepare state feature vector.
                state_dict = {
# Red horizontal.
                    "red_x": red_u['pos'][1], "red_y": red_u['pos'][0], 
# Blue horizontal.
                    "blue_x": blue_u['pos'][1], "blue_y": blue_u['pos'][0],
# Resource levels.
                    "red_hp": red_u['hp'], "red_ammo": red_u['ammo'], "red_fuel": red_u['fuel'], "red_morale": red_u['morale'],
# Spatial checks.
                    "red_elevation": self.env.elevation[red_u['pos'][0], red_u['pos'][1]],
# Distance check.
                    "distance": abs(red_u['pos'][0] - blue_u['pos'][0]) + abs(red_u['pos'][1] - blue_u['pos'][1]),
# Sensor visibility check.
                    "fog_visible": 1 if self.env.fog_map[red_u['pos'][0], red_u['pos'][1]] else 0,
# History check.
                    "red_prev_action": red_u.get('prev_action', 0)
# End of vector.
                }
# Execute Oracle state update.
                prediction = self.brain.oracle.update(state_dict)
# Empty line.

# Execute game logic ticks.
            # --- STEP TEAMS ---
# Process blue team moves via Tactical Brain.
            for i, b_unit in enumerate(self.env.blue_units):
# Only move living units.
                if b_unit['alive']: self.env.step(self.brain.get_smart_action(b_unit['id'], self.env), unit_index=i)
# Empty line.

# Process red team moves via Posture Engine.
            for r_unit in self.env.red_units:
# Only move living units.
                if r_unit['alive']:
# Run Red Bot Strategy decision.
                    r_act, _ = get_red_action(r_unit, self.env.blue_units, self.env.grid_size, self.env.terrain_map, red_all_units=self.env.red_units, supply_drops=self.env.supply_drops)
# Execute movement or idling.
                    if r_act <= 4: self.env._do_move(r_unit, r_act)
# Execute ranged attack.
                    elif r_act == 5: self.env._do_ranged_attack(r_unit)
# Track historical action.
                    r_unit['prev_action'] = r_act
# Empty line.

# Post-step world updates.
            self.env._rebuild_unit_map()
            self.env._update_fog()
# Empty line.

# Begin frame rendering sequence.
            # --- DRAW ---
# Layer 0: Background.
            self.screen.fill(COLOR_BG)
# Layer 1: Grid and Heatmaps.
            self._draw_grid(prediction)
# Layer 2: Units and Combat.
            self._draw_units()
# Layer 3: Side Dashboard.
            self._draw_hud(prediction)
# Empty line.
            
# Layer 4: CRT Post-process.
            # Post-Process: CRT Overlay
            self.screen.blit(self.scanline_surf, (0, 0))
# Empty line.

# Game termination check logic.
            # --- MATCH TERMINATION ---
# Scan for survivors on both sides.
            blue_alive = any(u['alive'] for u in self.env.blue_units)
            red_alive = any(u['alive'] for u in self.env.red_units)
# Check game over.
            if not blue_alive or not red_alive:
# Result string logic.
                winner = "BLUE WINS" if blue_alive else "RED WINS"
# Setup black fade surface.
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
# Semi-opaque black overlay.
                overlay.fill((0, 0, 0, 180))
                self.screen.blit(overlay, (0, 0))
# Render main winner text.
                txt = self.header_font.render(winner, True, (255, 255, 255))
                self.screen.blit(txt, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2))
# Final frame flip.
                pygame.display.flip()
# Pause for dramatic victory before reset.
                time.sleep(1.5)
# Reboot simulation, Brain, and Increment mission count.
                self.env.reset(); self.brain.reset(); self.match_count += 1
# Empty line.

# Physical frame update call.
            pygame.display.flip()
# Target frame rate control (1 FPS for tactical pacing).
            clock.tick(1)
# Empty line.

# Entry point condition check.
if __name__ == "__main__":
# Safe main execution call.
    UltimateCommander().run()
