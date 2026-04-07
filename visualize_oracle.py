"""
visualize_oracle.py — Cyber-Directive Edition (v17.0 RESTORED)
============================================================
The absolute peak of visual tactical discovery. 
Neon Glows | Animated Radar | Digital Filter | Live Health Overlays
Full Intelligence HUD | Intent Dashboard | Battle Log (Removed)
"""

import pygame
import sys
import copy
import time
import json
import random
import numpy as np

# TDASS System Imports
from battle_env import BattleEnv
from Element_Logic.red_strategy import get_red_action
from brain.tactical_brain import TacticalBrain

# ──────────────────────────────────────────────────────────────────────────
# 🛠️ CONFIGURATION & HUD DIMENSIONS
# ──────────────────────────────────────────────────────────────────────────
SCREEN_WIDTH  = 1240
SCREEN_HEIGHT = 800
GRID_SIZE     = 600
TILE_SIZE     = 60
HUD_X         = 680
HUD_WIDTH     = 540

# --- THEME: NEON TACTICAL ---
COLOR_BG      = (5, 8, 12)
COLOR_GRID    = (20, 35, 55)
COLOR_BLUE    = (0, 200, 255)
COLOR_RED     = (255, 30, 80)
COLOR_SUPPLY  = (255, 215, 0)
COLOR_INTEL   = (0, 255, 180)
COLOR_TEXT    = (200, 220, 240)

class UltimateCommander:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("☢️ TDSS CYBER-DIRECTIVE — RESTORED v17.0")
        
        # --- FONTS ---
        self.header_font = pygame.font.SysFont("Verdana", 26, bold=True)
        self.sub_font    = pygame.font.SysFont("Verdana", 18, bold=True)
        self.font        = pygame.font.SysFont("Verdana", 14)
        self.small_font  = pygame.font.SysFont("Consolas", 12)
        self.label_font  = pygame.font.SysFont("Consolas", 11, bold=True)

        # --- SIMULATION ---
        self.env = BattleEnv(render_mode=None)
        self.brain = TacticalBrain()
        self.env.reset()
        
        # --- STATE ---
        self.match_count = 1
        self.hover_pos = (0, 0)
        self.tick_count = 0
        self.scanline_offset = 0
        
        # --- HUD SURFACES ---
        self.scanline_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for y in range(0, SCREEN_HEIGHT, 4):
            pygame.draw.line(self.scanline_surf, (0, 0, 0, 45), (0, y), (SCREEN_WIDTH, y))

    # ────────────────────────────────────────────────────────────────────────
    # 🎨 RENDERING: BATTLEFIELD & NEON EFFECTS
    # ────────────────────────────────────────────────────────────────────────
    def _draw_grid(self, prediction):
        """Draw tactical 10x10 grid with animated Oracle heatmap."""
        pulse_alpha = abs(int(100 * np.sin(self.tick_count * 0.1))) + 50
        
        # Crosshair Highlight
        hr, hc = self.hover_pos
        pygame.draw.rect(self.screen, (10, 25, 45), (40, 100 + hr*60, 600, 60))
        pygame.draw.rect(self.screen, (10, 25, 45), (40 + hc*60, 100, 60, 600))

        for r in range(10):
            for c in range(10):
                t_type = self.env.terrain_map[r, c]
                color = (15, 20, 30) # Plains
                if t_type == 1: color = (50, 55, 70) # Wall
                if t_type == 2: color = (20, 45, 20) # Forest
                if t_type == 3: color = (10, 30, 80) # Water
                if t_type == 4: color = (70, 70, 80) # Road
                
                rect = (40 + c*60, 100 + r*60, 60, 60)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 1)

                # Heatmap Radar Sweep
                if prediction:
                    prob = prediction['heatmap'][r, c]
                    if prob > 0.08:
                        alpha = int(min(255, prob * pulse_alpha * 2.5))
                        overlay = pygame.Surface((58, 58), pygame.SRCALPHA)
                        overlay.fill((255, 215, 0, alpha // 4))
                        self.screen.blit(overlay, (41 + c*60, 101 + r*60))
                        # Animated '?'
                        if self.tick_count % 30 < 15:
                            q_txt = self.sub_font.render("?", True, (255, 255, 100))
                            self.screen.blit(q_txt, (60 + c*60, 115 + r*60))

        # Fog of War
        fog_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
        fog_surf.fill((0, 0, 5, 210))
        for r in range(10):
            for c in range(10):
                  if not self.env.fog_map[r, c]:
                      self.screen.blit(fog_surf, (40 + c*60, 100 + r*60))

        # Active Supply Drops
        for drop in self.env.supply_drops:
            dr, dc = drop['pos']
            pygame.draw.rect(self.screen, (255, 255, 255), (40 + dc*60+15, 100 + dr*60+15, 30, 30))
            pygame.draw.rect(self.screen, COLOR_SUPPLY, (40 + dc*60+18, 100 + dr*60+18, 24, 24))

    def _draw_units(self):
        """Render units with on-map Vitality bars and Neon Lasers."""
        for u in self.env.blue_units + self.env.red_units:
            if not u['alive']: continue
            r, c = u['pos']
            is_blue = (u['team'] == 'blue')
            
            if is_blue or self.env.fog_map[r, c]:
                center = (40 + c*60 + 30, 100 + r*60 + 30)
                main_color = COLOR_BLUE if is_blue else COLOR_RED
                
                # Glowing Unit Icon
                for i in range(3):
                    pygame.draw.circle(self.screen, main_color, center, 24-i*2, 1)
                pygame.draw.circle(self.screen, main_color, center, 18)
                
                # Floating HP Bar
                pygame.draw.rect(self.screen, (0, 0, 0), (40+c*60+5, 100+r*60+52, 50, 4))
                pygame.draw.rect(self.screen, (0, 255, 100), (40+c*60+5, 100+r*60+52, int(0.5 * u['hp']), 4))

                # Combat VFX: Neon Flicker Laser
                if u.get('prev_action') == 5:
                    opponents = self.env.red_units if is_blue else self.env.blue_units
                    living = [o for o in opponents if o['alive']]
                    if living:
                        tgt = min(living, key=lambda o: abs(r-o['pos'][0]) + abs(c-o['pos'][1]))
                        tr, tc = tgt['pos']
                        tgt_center = (40 + tc*60 + 30, 100 + tr*60 + 30)
                        if self.tick_count % 4 < 2:
                            pygame.draw.line(self.screen, (255, 255, 150), center, tgt_center, 6)
                            pygame.draw.line(self.screen, (255, 255, 255), center, tgt_center, 2)

    # ────────────────────────────────────────────────────────────────────────
    # 🖥️ HUD: TACTICAL INTELLIGENCE DASHBOARD
    # ────────────────────────────────────────────────────────────────────────
    def _draw_hud(self, prediction):
        """Render the high-density Oracle Intel panels."""
        # Panel Background
        pygame.draw.rect(self.screen, (10, 15, 25), (HUD_X, 20, HUD_WIDTH, 760), border_radius=10)
        pygame.draw.rect(self.screen, (30, 50, 80), (HUD_X, 20, HUD_WIDTH, 760), 2, border_radius=10)

        # 1. ORACLE PREDICTIVE INTENT
        y_off = 40
        self.screen.blit(self.header_font.render("ORACLE INTEL DASHBOARD", True, COLOR_INTEL), (HUD_X + 25, y_off))
        
        posture = prediction['posture'] if prediction else "INITIALIZING..."
        conf    = prediction['confidence'] if prediction else 0.0
        
        # Large Intent Card
        card_color = (255, 200, 0) if posture == "ATTACK" else (0, 200, 255)
        pygame.draw.rect(self.screen, (20, 30, 50), (HUD_X + 25, y_off + 50, 490, 100), border_radius=8)
        self.screen.blit(self.sub_font.render("PREDICTED ENEMY INTENT:", True, (150, 170, 190)), (HUD_X + 45, y_off + 65))
        self.screen.blit(self.header_font.render(f"[ {posture} ]", True, card_color), (HUD_X + 45, y_off + 95))
        
        # Confidence Bar
        self.screen.blit(self.small_font.render(f"ORACLE CONFIDENCE: {conf*100:.1f}%", True, COLOR_TEXT), (HUD_X + 25, y_off + 165))
        pygame.draw.rect(self.screen, (0, 0, 0), (HUD_X + 25, y_off + 185, 490, 12))
        pygame.draw.rect(self.screen, card_color, (HUD_X + 25, y_off + 185, int(490 * conf), 12))

        # 2. ENEMY THREAT SCANNER
        y_off = 260
        self.screen.blit(self.sub_font.render("🔴 ENEMY THREAT STATUS", True, COLOR_RED), (HUD_X + 25, y_off))
        
        for i, u in enumerate(self.env.red_units):
            row_y = y_off + 40 + (i * 70)
            pygame.draw.rect(self.screen, (15, 20, 35), (HUD_X+25, row_y, 490, 60), border_radius=5)
            
            r, c = u['pos']
            visible = self.env.fog_map[r, c]
            
            if not u['alive']:
                txt = "--- TARGET NEUTRALIZED ---"
                self.screen.blit(self.font.render(txt, True, (0, 255, 100)), (HUD_X + 150, row_y + 20))
            elif not visible:
                txt = "--- TARGET CLOAKED (IN FOG) ---"
                self.screen.blit(self.font.render(txt, True, (100, 110, 130)), (HUD_X + 140, row_y + 20))
            else:
                self.screen.blit(self.small_font.render(f"ID: {u['id']} | POS: [{c},{r}]", True, COLOR_TEXT), (HUD_X + 40, row_y + 12))
                # Mini HP Bar
                pygame.draw.rect(self.screen, (0, 0, 0), (HUD_X + 40, row_y + 35, 200, 8))
                pygame.draw.rect(self.screen, COLOR_RED, (HUD_X + 40, row_y + 35, int(2 * u['hp']), 8))
                self.screen.blit(self.small_font.render(f"AMMO: {u['ammo']}", True, (255, 150, 0)), (HUD_X + 260, row_y + 32))

        # 3. BATTLEFIELD SENSOR PROBE (Bottom Panel)
        y_off = 620
        self.screen.blit(self.sub_font.render("🔎 ORACLE RISK SIGNATURE", True, COLOR_INTEL), (HUD_X + 25, y_off))
        hr, hc = self.hover_pos
        t_type = self.env.terrain_map[hr, hc]
        t_names = ["PLAINS", "WALL", "FOREST", "WATER", "ROAD"]
        risk = prediction['heatmap'][hr, hc] if prediction else 0
        
        pygame.draw.rect(self.screen, (5, 30, 45), (HUD_X+25, y_off + 40, 490, 80), border_radius=5)
        self.screen.blit(self.font.render(f"PROBE COORDINATES: [{hc}, {hr}]", True, COLOR_TEXT), (HUD_X + 45, y_off + 50))
        self.screen.blit(self.font.render(f"TERRAIN: {t_names[t_type] if t_type < 5 else 'UNKNOWN'}", True, COLOR_TEXT), (HUD_X + 45, y_off + 70))
        self.screen.blit(self.font.render(f"LOCAL THREAT PROBABILITY: {risk*100:.2f}%", True, (255, 255, 0)), (HUD_X + 45, y_off + 90))

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            self.tick_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    if 40 < mx < 640 and 100 < my < 700:
                        self.hover_pos = ((my-100)//60, (mx-40)//60)

            # --- BRAIN: TACTICAL ORACLE ---
            red_u = next((u for u in self.env.red_units if u['alive']), None)
            blue_u = next((u for u in self.env.blue_units if u['alive']), None)
            prediction = None
            if red_u and blue_u:
                state_dict = {
                    "red_x": red_u['pos'][1], "red_y": red_u['pos'][0], "blue_x": blue_u['pos'][1], "blue_y": blue_u['pos'][0],
                    "red_hp": red_u['hp'], "red_ammo": red_u['ammo'], "red_fuel": red_u['fuel'], "red_morale": red_u['morale'],
                    "red_elevation": self.env.elevation[red_u['pos'][0], red_u['pos'][1]],
                    "distance": abs(red_u['pos'][0] - blue_u['pos'][0]) + abs(red_u['pos'][1] - blue_u['pos'][1]),
                    "fog_visible": 1 if self.env.fog_map[red_u['pos'][0], red_u['pos'][1]] else 0,
                    "red_prev_action": red_u.get('prev_action', 0)
                }
                prediction = self.brain.oracle.update(state_dict)

            # --- STEP TEAMS ---
            for i, b_unit in enumerate(self.env.blue_units):
                if b_unit['alive']: self.env.step(self.brain.get_smart_action(b_unit['id'], self.env), unit_index=i)

            for r_unit in self.env.red_units:
                if r_unit['alive']:
                    r_act, _ = get_red_action(r_unit, self.env.blue_units, self.env.grid_size, self.env.terrain_map, red_all_units=self.env.red_units, supply_drops=self.env.supply_drops)
                    if r_act <= 4: self.env._do_move(r_unit, r_act)
                    elif r_act == 5: self.env._do_ranged_attack(r_unit)
                    r_unit['prev_action'] = r_act

            self.env._rebuild_unit_map()
            self.env._update_fog()

            # --- DRAW ---
            self.screen.fill(COLOR_BG)
            self._draw_grid(prediction)
            self._draw_units()
            self._draw_hud(prediction)
            
            # Post-Process: CRT Overlay
            self.screen.blit(self.scanline_surf, (0, 0))

            # --- MATCH TERMINATION ---
            blue_alive = any(u['alive'] for u in self.env.blue_units)
            red_alive = any(u['alive'] for u in self.env.red_units)
            if not blue_alive or not red_alive:
                winner = "BLUE WINS" if blue_alive else "RED WINS"
                overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180))
                self.screen.blit(overlay, (0, 0))
                txt = self.header_font.render(winner, True, (255, 255, 255))
                self.screen.blit(txt, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2))
                pygame.display.flip()
                time.sleep(1.5)
                self.env.reset(); self.brain.reset(); self.match_count += 1

            pygame.display.flip()
            clock.tick(1)

if __name__ == "__main__":
    UltimateCommander().run()
