import pygame
import sys
import threading
import json
import torch
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from world import WorldState
import faulthandler

# Enable C++ segfault catching so the terminal tells us WHY it died
faulthandler.enable()

# Colors
BLACK = (15, 15, 20)
WHITE = (240, 240, 245)
GRAY = (100, 100, 110)
GREEN = (100, 255, 100)
BLUE = (100, 150, 255)
RED = (255, 100, 100)

class UniverseApp:
    def __init__(self):
        pygame.init()
        self.width = 1024
        self.height = 768
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Visual Universe Observer")
        
        self.font = pygame.font.SysFont("courier", 16)
        self.title_font = pygame.font.SysFont("courier", 24, bold=True)
        
        self.world = WorldState(device="cpu")
        
        self.running = True
        self.auto_evolve = False
        self.is_ingesting = False
        self.cancel_ingest = False
        self.last_metrics = None
        self.status_msg = "Ready."
        
        # Graph View State
        self.view_mode = "dream" # "dream" or "graph"
        self.graph_pos = {}
        self.graph_nodes = []
        self.graph_edges = []
        
        # Timeline slider state
        self.timeline_age = 0
        self.scrubbing_timeline = False
        self.history_snapshots = [] # List of tuples (age, node_id, py_image) to view
        
    def draw_text(self, text, x, y, color=WHITE, font=None):
        if font is None:
            font = self.font
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def update_graph_layout(self):
        """Builds a NetworkX graph and computes 2D coordinates using PCA on embeddings."""
        with self.world.lock:
            cursor = self.world.memory.conn.cursor()
            cursor.execute("SELECT id, parent_ids, generation, embedding FROM concepts")
            
            G = nx.DiGraph()
            
            embeddings = []
            node_ids = []
            
            for c in cursor:
                c_id, parent_ids_json, gen, emb_json = c
                emb = np.array(json.loads(emb_json)).flatten()
                    
                G.add_node(c_id, gen=gen)
                parent_ids = json.loads(parent_ids_json)
                for pid in parent_ids:
                    G.add_edge(pid, c_id)
                    
                embeddings.append(emb)
                node_ids.append(c_id)
        
        self.graph_nodes = list(G.nodes)
        self.graph_edges = list(G.edges)
        
        if len(embeddings) > 1:
            try:
                pca = PCA(n_components=2)
                coords = pca.fit_transform(embeddings)
                
                # Normalize coordinates to [-1, 1] range for drawing
                max_val = np.max(np.abs(coords))
                if max_val > 0:
                    coords = coords / max_val
                    
                self.graph_pos = {node_ids[i]: coords[i] for i in range(len(node_ids))}
            except Exception as e:
                print(f"PCA Layout failed: {e}")
                self.graph_pos = nx.spring_layout(G)
        elif len(embeddings) == 1:
            self.graph_pos = {node_ids[0]: np.array([0.0, 0.0])}
        else:
            self.graph_pos = {}

    def draw_graph(self):
        """Draws the concept graph on the right panel."""
        with self.world.lock:
            # Center of the right panel: x=480+256=736, y=80+256=336
            # Adjust graph center slightly up to make room for timeline slider
            cx, cy = 736, 310
            scale = 220 # Scale coordinates from [-1, 1] to pixel space
            
            # Generate species colors
            species_colors = {}
            for node_id in self.graph_nodes:
                if node_id in self.world.ecologist.node_to_species:
                    s_id = self.world.ecologist.node_to_species[node_id]
                    if s_id not in species_colors:
                        # Hash species id to a color
                        np.random.seed(s_id)
                        species_colors[s_id] = (np.random.randint(50, 255), 
                                                np.random.randint(50, 255), 
                                                np.random.randint(50, 255))
            
            # Draw edges
            for edge in self.graph_edges:
                if edge[0] in self.graph_pos and edge[1] in self.graph_pos:
                    p1 = self.graph_pos[edge[0]]
                    p2 = self.graph_pos[edge[1]]
                    x1, y1 = int(p1[0] * scale + cx), int(p1[1] * scale + cy)
                    x2, y2 = int(p2[0] * scale + cx), int(p2[1] * scale + cy)
                    pygame.draw.line(self.screen, GRAY, (x1, y1), (x2, y2), 1)
            
            # Draw nodes
            for node in self.graph_nodes:
                if node in self.graph_pos:
                    p = self.graph_pos[node]
                    x, y = int(p[0] * scale + cx), int(p[1] * scale + cy)
                    
                    color = BLUE
                    # Color by species if it belongs to one
                    if node in self.world.ecologist.node_to_species:
                        s_id = self.world.ecologist.node_to_species[node]
                        if s_id in species_colors:
                            color = species_colors[s_id]

                    radius = 4
                    if node in self.world.active_concepts:
                        color = GREEN
                        radius = 6
                        
                    pygame.draw.circle(self.screen, color, (x, y), radius)
                    
            # Draw small image preview at bottom right
            if self.world.current_image:
                mode = self.world.current_image.mode
                size = self.world.current_image.size
                data = self.world.current_image.tobytes()
                py_image = pygame.image.fromstring(data, size, mode)
                # Scale down to 128x128 thumbnail
                py_image = pygame.transform.scale(py_image, (128, 128))
                # Draw border
                pygame.draw.rect(self.screen, GRAY, (864, 464, 128, 128), 1)
                # Draw image in bottom right corner of the right panel
                self.screen.blit(py_image, (864, 464))

    def draw_ui(self):
        with self.world.lock:
            self.screen.fill(BLACK)
            
            # Title
            self.draw_text("VISUAL UNIVERSE LAB", 20, 20, BLUE, self.title_font)
            
            # Left Panel: Metrics & Info
            self.draw_text(f"Age: {self.world.age}", 20, 80)
            
            # Concept Graph Status
            graph_status = self.world.concept_graph.get_status()
            total_nodes = graph_status['total_nodes']
            reality_nodes = graph_status['reality_nodes']
            concept_nodes = graph_status['concept_nodes']
            dream_nodes = graph_status['dream_nodes']
            
            self.draw_text(f"Graph: {total_nodes} nodes", 20, 110)
            self.draw_text(f"Reality: {reality_nodes}", 20, 140, GREEN)
            self.draw_text(f"Concept: {concept_nodes}", 20, 170, BLUE)
            self.draw_text(f"Dream: {dream_nodes}", 20, 200, GRAY)
            
            # Graph promotions/quarantines
            self.draw_text(f"Promoted: {graph_status['total_promotions']}", 20, 230)
            self.draw_text(f"Quarantined/Extinct: {graph_status['total_quarantines']}", 20, 260)
            
            # GNN influence
            gnn_influence_pct = int(self.world.gnn_influence * 100)
            self.draw_text(f"GNN Influence: {gnn_influence_pct}%", 20, 300, 
                          GREEN if gnn_influence_pct > 60 else BLUE if gnn_influence_pct > 40 else GRAY)
            
            if self.last_metrics:
                mode = self.last_metrics.get('mode', 'unknown')
                self.draw_text(f"Last Step: {mode}", 20, 370, BLUE)
                self.draw_text(f"Reward: {self.last_metrics.get('total_reward', 0):.3f}", 20, 400)
                
                # NEW: Prediction error (learning signal!)
                pred_error = self.last_metrics.get('prediction_error', 0)
                conf = self.last_metrics.get('confidence', 0)
                tier = self.last_metrics.get('tier', 'unknown')
                tags = self.last_metrics.get('tags', [])
                
                self.draw_text(f"Pred Error: {pred_error:.3f}", 20, 430)
                self.draw_text(f"Confidence: {conf:.2f}", 20, 460)
                self.draw_text(f"Tier: {tier}", 20, 490, 
                              GREEN if tier == 'landmark' else BLUE if tier == 'working' else GRAY)
                
                if tags:
                    self.draw_text(f"Tags: {', '.join(tags)}", 20, 520, GREEN)
                
                # Draw Physics / Ecology Info
                active_id = self.world.active_concepts[0] if self.world.active_concepts else None
                y = 560
                
                if active_id and active_id in self.world.ecologist.node_to_species:
                    s_id = self.world.ecologist.node_to_species[active_id]
                    if s_id in self.world.ecologist.species:
                        s = self.world.ecologist.species[s_id]
                        self.draw_text(f"Species: {s.id} ({s.tier.value})", 20, y, GREEN)
                        y += 30
                        self.draw_text(f"Culture biases:", 20, y, GRAY)
                        y += 30
                        for name, bias in s.culture.aesthetic_biases.items():
                            self.draw_text(f"  {name}: {bias:.2f}", 20, y, GRAY)
                            y += 30
                else:
                    self.draw_text(f"Species: None (Rogue/Orphan)", 20, y, GRAY)
                    y += 30
                    
                # Multi-head RL weights (adaptive!)
                weights = self.last_metrics.get('weights', {})
                if weights:
                    self.draw_text("RL Weights:", 20, y, GREEN)
                    y += 30
                    for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]:
                        self.draw_text(f"{name[:8]:8s} {weight:.2f}", 20, y)
                        y += 30
                
            # Controls (moved down to avoid overlap)
            self.draw_text("Controls:", 250, 680, GRAY)
            self.draw_text("[SPACE] Step Evolution", 250, 710)
            self.draw_text("[A] Toggle Auto-Evolve", 250, 740)
            self.draw_text("[I] Ingest Dataset / Cancel", 250, 770)
            self.draw_text("[V] Toggle View (Dream/Graph)", 250, 800)
            
            self.draw_text(f"Auto-Evolve: {'ON' if self.auto_evolve else 'OFF'}", 20, 840, GREEN if self.auto_evolve else GRAY)
            self.draw_text(f"Ingesting: {'ON' if self.is_ingesting else 'OFF'}", 20, 870, GREEN if self.is_ingesting else GRAY)
            self.draw_text(f"Current View: {self.view_mode.upper()}", 20, 900, BLUE)
            
            if hasattr(self.world.meta_controller, 'attention_energy') and self.world.meta_controller.attention_energy > 0.01:
                self.draw_text(f"Attention Energy: {self.world.meta_controller.attention_energy:.2f}", 20, 920, RED)
                
            # Status at bottom
            self.draw_text(f"Status: {self.status_msg}", 20, 950, BLUE)
            
            # Right Panel: Visualization Area
            pygame.draw.rect(self.screen, (30, 30, 35), (480, 80, 512, 512))
            
            if self.view_mode == "dream":
                if self.world.current_image:
                    mode = self.world.current_image.mode
                    size = self.world.current_image.size
                    data = self.world.current_image.tobytes()
                    py_image = pygame.image.fromstring(data, size, mode)
                    py_image = pygame.transform.scale(py_image, (512, 512))
                    self.screen.blit(py_image, (480, 80))
            elif self.view_mode == "graph":
                self.draw_graph()
                
            # Timeline Slider (Bottom Center)
            self.draw_timeline()
            
            pygame.display.flip()
            
    def load_timeline_snapshot(self):
        """Loads a historical image at self.timeline_age."""
        with self.world.lock:
            # Closest concept by generation. Evolved concepts may have NULL latent
            # (generated on demand) — never json.loads a None.
            cursor = self.world.memory.conn.cursor()
            cursor.execute(
                "SELECT id, latent FROM concepts ORDER BY ABS(generation - ?) LIMIT 1",
                (self.timeline_age,),
            )
            res = cursor.fetchone()
            if not res:
                return
            concept_id, latent_json = res
            try:
                if latent_json:
                    latent_data = json.loads(latent_json)
                    latent_tensor = torch.tensor(latent_data, dtype=torch.float32).to(self.world.device)
                    if latent_tensor.dim() == 3:
                        latent_tensor = latent_tensor.unsqueeze(0)
                else:
                    latent_tensor = self.world.get_latent_for_concept(concept_id)
                if latent_tensor is None:
                    return
                self.world.current_image = self.world.vision.decode_latent_to_image(latent_tensor)
            except Exception as e:
                print(f"[Timeline] Failed to load gen~{self.timeline_age} (id={concept_id}): {e}")

    def draw_timeline(self):
        """Draw a timeline slider at the bottom of the right panel."""
        cx = 480
        cy = 620
        width = 512
        height = 40
        pygame.draw.rect(self.screen, (40, 40, 45), (cx, cy, width, height))
        
        # Max age
        max_age = max(1, self.world.age)
        
        # Draw line
        pygame.draw.line(self.screen, GRAY, (cx + 20, cy + 20), (cx + width - 20, cy + 20), 2)
        
        # Draw current age marker
        handle_x = cx + 20 + int((self.timeline_age / max_age) * (width - 40))
        pygame.draw.circle(self.screen, BLUE, (handle_x, cy + 20), 6)
        
        self.draw_text(f"Timeline Scrub: Gen {self.timeline_age}", cx + 20, cy + 45, BLUE)

    def evolve_step(self):
        self.status_msg = "Dreaming..."
        # DO NOT call self.draw_ui() from background thread (causes Pygame silent crash)
        try:
            success, metrics = self.world.step()
        except Exception as e:
            print(f"[Evolve] step crashed (kept alive): {e}")
            import traceback
            traceback.print_exc()
            self.status_msg = f"Evolve error: {e}"
            return
        self.last_metrics = metrics
        if success:
            self.status_msg = "New concept accepted."
            # Only auto-update timeline age if we are not actively scrubbing
            if not self.scrubbing_timeline and self.world.age > self.timeline_age:
                self.timeline_age = self.world.age
                
            # Update graph if we are currently looking at it
            if self.view_mode == "graph":
                self.update_graph_layout()
        else:
            self.status_msg = "Concept rejected by Reality Anchor."

    def ingest_dataset(self):
        import os
        self.is_ingesting = True
        self.cancel_ingest = False
        self.status_msg = "Scanning dataset folder..."
        
        dataset_dir = "dataset"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
        files = [f for f in os.listdir(dataset_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        
        import random
        random.shuffle(files)
        
        success_count = 0
        for i, filename in enumerate(files):
            if self.cancel_ingest:
                self.status_msg = f"Ingestion cancelled. Added {success_count} images."
                break
                
            self.status_msg = f"Ingesting {i+1}/{len(files)}: {filename[:15]}..."
            
            with self.world.lock:
                if self.world.memory.is_file_ingested(filename):
                    continue
                
            filepath = os.path.join(dataset_dir, filename)
            success, c_id = self.world.ingest_image(filepath)
            if success and c_id is not None:
                with self.world.lock:
                    self.world.memory.mark_file_ingested(filename, c_id)
                success_count += 1
                
        else:
            self.status_msg = f"Ingestion complete. Added {success_count} new images."
            
        self.is_ingesting = False
        if self.view_mode == "graph":
            self.update_graph_layout()

    def handle_click(self, pos):
        """Handle mouse clicks in graph view to inject attention or scrub timeline."""
        click_x, click_y = pos
        
        # Timeline clicking
        cx, cy, width, height = 480, 620, 512, 40
        if cx <= click_x <= cx + width and cy <= click_y <= cy + height + 20: # +20 for text
            max_age = max(1, self.world.age)
            relative_x = np.clip(click_x - (cx + 20), 0, width - 40)
            self.timeline_age = int((relative_x / (width - 40)) * max_age)
            self.load_timeline_snapshot()
            self.status_msg = f"Scrubbing timeline to Gen {self.timeline_age}..."
            self.draw_ui()
            return
            
        if self.view_mode != "graph" or not self.graph_pos:
            return
            
        cx, cy = 736, 310
        scale = 220
        
        # Find nearest node
        best_node = None
        min_dist = float('inf')
        
        for node, p in self.graph_pos.items():
            nx, ny = int(p[0] * scale + cx), int(p[1] * scale + cy)
            dist = (click_x - nx)**2 + (click_y - ny)**2
            if dist < min_dist:
                min_dist = dist
                best_node = node
                
        # If click was close enough to a node (radius 10 pixels -> squared 100)
        if min_dist < 100 and best_node is not None:
            if self.world.inject_attention(best_node):
                self.status_msg = f"Injected attention into Concept {best_node}"
                self.draw_ui()

    def run(self):
        clock = pygame.time.Clock()
        
        evolve_thread = None
        ingest_thread = None
        background_ingest_timer = 0
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click
                        self.handle_click(event.pos)
                        self.scrubbing_timeline = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.scrubbing_timeline = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.scrubbing_timeline:
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if evolve_thread is None or not evolve_thread.is_alive():
                            evolve_thread = threading.Thread(target=self.evolve_step)
                            evolve_thread.start()
                    elif event.key == pygame.K_a:
                        self.auto_evolve = not self.auto_evolve
                    elif event.key == pygame.K_i:
                        if self.is_ingesting:
                            self.cancel_ingest = True
                        else:
                            if ingest_thread is None or not ingest_thread.is_alive():
                                ingest_thread = threading.Thread(target=self.ingest_dataset)
                                ingest_thread.start()
                    elif event.key == pygame.K_v:
                        if self.view_mode == "dream":
                            self.view_mode = "graph"
                            self.status_msg = "Calculating graph layout..."
                            self.draw_ui()
                            self.update_graph_layout()
                            self.status_msg = "Graph view active."
                        else:
                            self.view_mode = "dream"
                            self.status_msg = "Dream view active."
                        
            if self.auto_evolve:
                if evolve_thread is None or not evolve_thread.is_alive():
                    evolve_thread = threading.Thread(target=self.evolve_step)
                    evolve_thread.start()
                    
            # Background slow ingestion
            background_ingest_timer += 1
            if background_ingest_timer >= 60 and not self.is_ingesting:  # Try every ~2 seconds
                background_ingest_timer = 0
                if ingest_thread is None or not ingest_thread.is_alive():
                    def bg_task():
                        self.world.background_ingest_step()
                    ingest_thread = threading.Thread(target=bg_task)
                    ingest_thread.start()
                    
            self.draw_ui()
            clock.tick(30)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = UniverseApp()
    app.run()
