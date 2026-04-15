# The primary visual command and control (C2) interface for the TDASS simulation.
"""
# This script integrates all subsystems into a real-time, high-fidelity tactical dashboard.
visualize_oracle.py — Cyber-Directive Edition (v17.0 RESTORED)
============================================================
# It provides a 'Holographic' view of the battlefield, combining live unit telemetry with
# real-time AI 'Oracle' predictions and spatial risk heatmaps.

Key Features:
- Neon Aesthetic HUD: High-density data readouts for tactical decision-making.
- Oracle Heatmap Overlay: Visualizes the enemy's most likely future positions.
- Intent Awareness Card: Real-time classification of adversary posture (Attack, Retreat, etc.).
- Combat VFX: Dynamic laser and projectile line rendering for engagement tracking.
"""

# Pygame for high-performance 2D hardware-accelerated rendering.
import pygame
import sys
import copy
import time
import json
import random
import numpy as np

# Internal Logic Imports
from battle_env import BattleEnv
from bread.tactical_brain import TacticalBrain # Ensure correctly mapped in workspace
from Element_Logic.red_strategy import get_red_action

# ──────────────────────────────────────────────────────────────────────────
# Configuration & HUD Geometry
# ──────────────────────────────────────────────────────────────────────────

SCREEN_WIDTH  = 1240
SCREEN_HEIGHT = 800
GRID_SIZE     = 600
TILE_SIZE     = 60
HUD_X         = 680
HUD_WIDTH     = 540

# ── Theme: Cyber-Directive Neon ──
COLOR_BG      = (5, 8, 12)    # Void-Black
COLOR_GRID    = (20, 35, 55)  # Muted Wireframe
COLOR_BLUE    = (0, 200, 255) # Friendly Neon
COLOR_RED     = (255, 30, 80) # Hostile Alert
COLOR_SUPPLY  = (255, 215, 0) # Logistics Gold
COLOR_INTEL   = (0, 255, 180) # Oracle Green
COLOR_TEXT    = (200, 220, 240)# Cyber-White

class UltimateCommander:
    """
    Top-level application orchestrator. 
    Manages the Pygame loop, input handling, and subsystem rendering.
    """

    def __init__(self):
        """Initializes hardware surfaces, typography, and tactical logic engines."""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("☢️ TDSS CYBER-DIRECTIVE — COMMAND INTERFACE")
        
        # Typography: Using system fonts to create a technical/UI look.
        self.header_font = pygame.font.SysFont("Verdana", 26, bold=True)
        self.sub_font    = pygame.font.SysFont("Verdana", 18, bold=True)
        self.font        = pygame.font.SysFont("Verdana", 14)
        self.small_font  = pygame.font.SysFont("Consolas", 12)
        self.label_font  = pygame.font.SysFont("Consolas", 11, bold=True)

        # Tactical Simulation Initialization
        # We run the environment in 'None' render mode to bypass the default Gymnasium viewer.
        self.env = BattleEnv(render_mode=None)
        # The 'TacticalBrain' hosts the Oracle and Decision Feasibility logic.
        self.brain = TacticalBrain()
        self.env.reset()
        
        # State Tracking
        self.match_count = 1
        self.hover_pos = (0, 0) # Synchronized with tile the user is 'Probing'.
        self.tick_count = 0     # Driven by Pygame clock for animations.
        
        # Post-Process: CRT Scanline Layer
        self.scanline_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for y in range(0, SCREEN_HEIGHT, 4):
            pygame.draw.line(self.scanline_surf, (0, 0, 0, 45), (0, y), (SCREEN_WIDTH, y))

    # ────────────────────────────────────────────────────────────────────────
    # 🎨 Rendering: Battlefield & AI Overlays
    # ────────────────────────────────────────────────────────────────────────

    def _draw_grid(self, prediction):
        """Renders the battlefield tiles and the animated AI risk heatmap."""
        # Calculate a pulse effect for the AI labels (Hacker-style breathing).
        pulse_alpha = abs(int(100 * np.sin(self.tick_count * 0.1))) + 50
        
        # UI: Crosshair Highlight based on Mouse Hover.
        hr, hc = self.hover_pos
        pygame.draw.rect(self.screen, (10, 25, 45), (40, 100 + hr*60, 600, 60)) # Row Shadow
        pygame.draw.rect(self.screen, (10, 25, 45), (40 + hc*60, 100, 60, 600)) # Col Shadow

        # Layer 1: Base Terrain Grid.
        for r in range(10):
            for c in range(10):
                t_type = self.env.terrain_map[r, c]
                # Map Terrain IDs to specific aesthetic color blocks.
                color = (15, 20, 30) # Plains
                if t_type == 1: color = (50, 55, 70) # Wall (Heavy Gray)
                if t_type == 2: color = (20, 45, 20) # Forest (Deep Green)
                if t_type == 3: color = (10, 30, 80) # Water (Tactical Navy)
                if t_type == 4: color = (70, 70, 80) # Road (High-contrast Slate)
                
                rect = (40 + c*60, 100 + r*60, 60, 60)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 1) # Wireframe effect

                # Layer 2: Oracle Predictive Intel Heatmap.
                # Here we visualize the 'Heat' from the LSTM's spatial probability grid.
                if prediction:
                    prob = prediction['heatmap'][r, c]
                    if prob > 0.08: # Intensity Threshold for visibility.
                        alpha = int(min(255, prob * pulse_alpha * 2.5))
                        # Render the 'Peril Zone' highlight.
                        overlay = pygame.Surface((58, 58), pygame.SRCALPHA)
                        overlay.fill((255, 215, 0, alpha // 4)) 
                        self.screen.blit(overlay, (41 + c*60, 101 + r*60))
                        
                        # Add a '?' indicator for areas of high tactical uncertainty.
                        if self.tick_count % 30 < 15:
                            q_txt = self.sub_font.render("?", True, (255, 255, 100))
                            self.screen.blit(q_txt, (60 + c*60, 115 + r*60))

        # Layer 3: Fog of War Simulation.
        # Areas not visible to Blue sensors are shrouded in a semi-opaque navy veil.
        fog_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
        fog_surf.fill((0, 0, 5, 210))
        for r in range(10):
            for c in range(10):
                  if not self.env.fog_map[r, c]:
                      self.screen.blit(fog_surf, (40 + c*60, 100 + r*60))

        # Layer 4: Logistics Checkpoints (Crates).
        for drop in self.env.supply_drops:
            dr, dc = drop['pos']
            pygame.draw.rect(self.screen, (255, 255, 255), (40 + dc*60+15, 100 + dr*60+15, 30, 30))
            pygame.draw.rect(self.screen, COLOR_SUPPLY, (40 + dc*60+18, 100 + dr*60+18, 24, 24))

    def _draw_units(self):
        """Renders tactical agents and dynamic combat visual effects (VFX)."""
        for u in self.env.blue_units + self.env.red_units:
            if not u['alive']: continue
            r, c = u['pos']
            is_blue = (u['team'] == 'blue')
            
            # Visibility Threshold: Units are only drawn if friendly OR in line-of-sight.
            if is_blue or self.env.fog_map[r, c]:
                center = (40 + c*60 + 30, 100 + r*60 + 30)
                main_color = COLOR_BLUE if is_blue else COLOR_RED
                
                # Holographic Unit Icon (Expanding Concentric Rings).
                for i in range(3):
                    pygame.draw.circle(self.screen, main_color, center, 24-i*2, 1)
                pygame.draw.circle(self.screen, main_color, center, 18)
                
                # Dynamic Vitality Overlays.
                pygame.draw.rect(self.screen, (0, 0, 0), (40+c*60+5, 100+r*60+52, 50, 4))
                pygame.draw.rect(self.screen, (0, 255, 100), (40+c*60+5, 100+r*60+52, int(0.5 * u['hp']), 4))

                # Combat VFX: Recoil/Laser tracer for Ranged Attacks.
                if u.get('prev_action') == 5:
                    opponents = self.env.red_units if is_blue else self.env.blue_units
                    living = [o for o in opponents if o['alive']]
                    if living:
                        # Draw a high-intensity laser tracer from source to target.
                        tgt = min(living, key=lambda o: abs(r-o['pos'][0]) + abs(c-o['pos'][1]))
                        tc, tr = tgt['pos'][1], tgt['pos'][0]
                        tgt_center = (40 + tc*60 + 30, 100 + tr*60 + 30)
                        if self.tick_count % 4 < 2:
                            pygame.draw.line(self.screen, (255, 255, 150), center, tgt_center, 6) # Glow
                            pygame.draw.line(self.screen, (255, 255, 255), center, tgt_center, 2) # Core

    # ────────────────────────────────────────────────────────────────────────
    # 🖥️ HUD: Strategic Dashboard
    # ────────────────────────────────────────────────────────────────────────

    def _draw_hud(self, prediction):
        """Displays the high-resolution Intelligence Panel (Side Bar)."""
        # Master Panel Foundation.
        pygame.draw.rect(self.screen, (10, 15, 25), (HUD_X, 20, HUD_WIDTH, 760), border_radius=10)
        pygame.draw.rect(self.screen, (30, 50, 80), (HUD_X, 20, HUD_WIDTH, 760), 2, border_radius=10)

        # block 1: Oracle Intent & Confidence Meter.
        y_off = 40
        self.screen.blit(self.header_font.render("ORACLE TACTICAL INTEL", True, COLOR_INTEL), (HUD_X + 25, y_off))
        
        posture = prediction['posture'] if prediction else "ANALYZING..."
        conf    = prediction['confidence'] if prediction else 0.0
        
        # Primary Intent Result Card.
        card_color = (255, 200, 0) if posture == "ATTACK" else (0, 200, 255)
        pygame.draw.rect(self.screen, (20, 30, 50), (HUD_X + 25, y_off + 50, 490, 100), border_radius=8)
        self.screen.blit(self.sub_font.render("PREDICTED ENEMY STRATEGY:", True, (150, 170, 190)), (HUD_X + 45, y_off + 65))
        self.screen.blit(self.header_font.render(f"[ {posture} ]", True, card_color), (HUD_X + 45, y_off + 95))
        
        # Reliability Bar (Confidence in prediction).
        self.screen.blit(self.small_font.render(f"ORACLE RELIABILITY INDEX: {conf*100:.1f}%", True, COLOR_TEXT), (HUD_X + 25, y_off + 165))
        pygame.draw.rect(self.screen, (0, 0, 0), (HUD_X + 25, y_off + 185, 490, 12))
        pygame.draw.rect(self.screen, card_color, (HUD_X + 25, y_off + 185, int(490 * conf), 12))

        # block 2: Live Adversary Status.
        y_off = 260
        self.screen.blit(self.sub_font.render("🔴 ADVERSARY THREAT SCANNER", True, COLOR_RED), (HUD_X + 25, y_off))
        for i, u in enumerate(self.env.red_units):
            row_y = y_off + 40 + (i * 70)
            pygame.draw.rect(self.screen, (15, 20, 35), (HUD_X+25, row_y, 490, 60), border_radius=5)
            
            r, c = u['pos']
            visible = self.env.fog_map[r, c]
            
            if not u['alive']:
                self.screen.blit(self.font.render("--- TARGET NEUTRALIZED ---", True, (0, 255, 100)), (HUD_X + 150, row_y + 20))
            elif not visible:
                self.screen.blit(self.font.render("--- TARGET CLOAKED (OFF-RADAR) ---", True, (100, 110, 130)), (HUD_X + 140, row_y + 20))
            else:
                self.screen.blit(self.small_font.render(f"ID: {u['id']} | COORDINATES: [{c},{r}]", True, COLOR_TEXT), (HUD_X + 40, row_y + 12))
                pygame.draw.rect(self.screen, (0, 0, 0), (HUD_X + 40, row_y + 35, 200, 8))
                pygame.draw.rect(self.screen, COLOR_RED, (HUD_X + 40, row_y + 35, int(2 * u['hp']), 8))
                self.screen.blit(self.small_font.render(f"AMMO: {u['ammo']}", True, (255, 150, 0)), (HUD_X + 260, row_y + 32))

        # block 3: Localized Sensor Probe (Mouse Proximity).
        y_off = 620
        self.screen.blit(self.sub_font.render("🔎 GRID SENSOR READOUT", True, COLOR_INTEL), (HUD_X + 25, y_off))
        hr, hc = self.hover_pos
        t_type = self.env.terrain_map[hr, hc]
        t_names = ["PLAINS", "WALL", "FOREST", "WATER", "ROAD"]
        risk = prediction['heatmap'][hr, hc] if prediction else 0
        
        pygame.draw.rect(self.screen, (5, 30, 45), (HUD_X+25, y_off + 40, 490, 80), border_radius=5)
        self.screen.blit(self.font.render(f"PROBE COORDINATES: [{hc}, {hr}]", True, COLOR_TEXT), (HUD_X + 45, y_off + 50))
        self.screen.blit(self.font.render(f"TERRAIN ANALYSIS: {t_names[t_type] if t_type < 5 else 'UNKNOWN'}", True, COLOR_TEXT), (HUD_X + 45, y_off + 70))
        self.screen.blit(self.font.render(f"ORACLE RISK QUOTIENT: {risk*100:.2f}%", True, (255, 255, 0)), (HUD_X + 45, y_off + 90))

    def run(self):
        """Standard Game Loop: Handle Events -> AI Inference -> Logic -> Draw."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            self.tick_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    # Grid Mapping: Convert pixel coordinates to 0-9 indices.
                    if 40 < mx < 640 and 100 < my < 700:
                        self.hover_pos = ((my-100)//60, (mx-40)//60)

            # ── AI Pass: Perception Update ──
            # We call the TacticalBrain to refresh the Oracle's LSTM state based on latest telemetry.
            red_u = next((u for u in self.env.red_units if u['alive']), None)
            blue_u = next((u for u in self.env.blue_units if u['alive']), None)
            prediction = None
            if red_u and blue_u:
                # State Vector Assembly for the Oracle.
                state_dict = {
                    "red_x": red_u['pos'][1], "red_y": red_u['pos'][0], 
                    "blue_x": blue_u['pos'][1], "blue_y": blue_u['pos'][0],
                    "red_hp": red_u['hp'], "red_ammo": red_u['ammo'], "red_fuel": red_u['fuel'], "red_morale": red_u['morale'],
                    "red_elevation": self.env.elevation[red_u['pos'][0], red_u['pos'][1]],
                    "distance": abs(red_u['pos'][0] - blue_u['pos'][0]) + abs(red_u['pos'][1] - blue_u['pos'][1]),
                    "fog_visible": 1 if self.env.fog_map[red_u['pos'][0], red_u['pos'][1]] else 0,
                    "red_prev_action": red_u.get('prev_action', 0)
                }
                # Sync the Oracle logic.
                prediction = self.brain.oracle.update(state_dict)

            # ── Logic Pass: Tactical Execution ──
            # Step Blue Team (Driven by the 'Smart' TacticalBrain).
            for i, b_unit in enumerate(self.env.blue_units):
                if b_unit['alive']: 
                    self.env.step(self.brain.get_smart_action(b_unit['id'], self.env), unit_index=i)

            # Step Red Team (Driven by the 'Heuristic' Teacher Strategy).
            for r_unit in self.env.red_units:
                if r_unit['alive']:
                    r_act, _ = get_red_action(r_unit, self.env.blue_units, self.env.grid_size, self.env.terrain_map, 
                                              red_all_units=self.env.red_units, supply_drops=self.env.supply_drops)
                    if r_act <= 4: self.env._do_move(r_unit, r_act)
                    elif r_act == 5: self.env._do_ranged_attack(r_unit)
                    r_unit['prev_action'] = r_act

            # Physical Sync.
            self.env._rebuild_unit_map()
            self.env._update_fog()

            # ── Render Pass: Composite Scene ──
            self.screen.fill(COLOR_BG)
            self._draw_grid(prediction)
            self._draw_units()
            self._draw_hud(prediction)
            
            # Application of the CRT Scanline Post-process.
            self.screen.blit(self.scanline_surf, (0, 0))

            # Match Resolution Check.
            blue_alive = any(u['alive'] for u in self.env.blue_units)
            red_alive = any(u['alive'] for u in self.env.red_units)
            if not blue_alive or not red_alive:
                winner = "BLUE TEAM DOMINANT" if blue_alive else "RED TEAM DOMINANT"
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180)) # Fade to Black.
                self.screen.blit(overlay, (0, 0))
                txt = self.header_font.render(winner, True, (255, 255, 255))
                self.screen.blit(txt, (SCREEN_WIDTH // 2 - 180, SCREEN_HEIGHT // 2))
                pygame.display.flip()
                time.sleep(2.0)
                # Mission Reboot Sequence.
                self.env.reset(); self.brain.reset(); self.match_count += 1

            pygame.display.flip()
            # Tactical Pacing: Locked to 1 frame per second for optimal human analysis.
            clock.tick(1)

if __name__ == "__main__":
    UltimateCommander().run()
