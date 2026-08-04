import os
# Limit native BLAS/OpenMP threads before PyTorch initializes (Windows CPU safety)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pygame
import sys
import threading
import time
import json
import torch
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from world import WorldState
import faulthandler

# Enable C++ segfault catching so the terminal tells us WHY it died
faulthandler.enable()

# PyTorch C++ backend is not thread-safe on CPU
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# Colors
BLACK = (15, 15, 20)
WHITE = (240, 240, 245)
GRAY = (100, 100, 110)
GREEN = (100, 255, 100)
BLUE = (100, 150, 255)
RED = (255, 100, 100)
ORANGE = (255, 170, 80)
PURPLE = (180, 120, 255)

# Graph layer filters (cycle with [T] while in graph view)
GRAPH_FILTERS = ("combined", "reality", "dreams")
TIER_COLORS = {
    "reality": GREEN,
    "concept": BLUE,
    "dream": ORANGE,
}

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
        
        # Start online trainer
        from training_scheduler import OnlineTrainer
        self.trainer = OnlineTrainer(self.world.model, self.world.memory, device="cpu")
        self.trainer.start()
        
        self.running = True
        self.auto_evolve = False
        self.is_ingesting = False
        self.cancel_ingest = False
        self.last_metrics = None
        self.status_msg = "Ready."
        self._evolve_lock = threading.Lock()
        self._auto_evolve_thread = None
        self._manual_evolve_thread = None
        self.evolve_attempts = 0
        self.evolve_accepted = 0
        self._evolve_busy = False
        
        # Graph View State
        self.view_mode = "dream"  # "dream" or "graph"
        self.graph_filter = "combined"  # "combined" | "reality" | "dreams"
        self.graph_pos = {}
        self.graph_nodes = []
        self.graph_edges = []
        self.graph_tiers = {}  # node_id -> "reality"|"concept"|"dream"
        
        # Timeline slider state
        self.timeline_age = 0
        self.scrubbing_timeline = False
        self.history_snapshots = [] # List of tuples (age, node_id, py_image) to view

        # UI perf caches
        self._frame = 0
        self._cached_dream_surface = None
        self._cached_dream_key = None
        self._graph_dirty = True
        self._graph_layout_age = -1
        self._graph_layout_interval = 25
        self._graph_max_pca_fit = 2000
        self._last_snap = None
        self._empty_snap = {
            "age": 0,
            "active_concepts": set(),
            "current_image": None,
            "attention_energy": 0.0,
            "gnn_influence": 0.0,
            "ecologist": None,
            "energy_pool": 0.0,
            "max_energy": 5000,
            "reality_live": 0,
            "dream_live": 0,
            "concept_live": 0,
            "graph_status": {},
        }
    def draw_text(self, text, x, y, color=WHITE, font=None):
        if font is None:
            font = self.font
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def _node_passes_filter(self, tier_value: str) -> bool:
        """Whether a concept-graph tier belongs in the current graph filter."""
        if self.graph_filter == "combined":
            return True
        if self.graph_filter == "reality":
            return tier_value == "reality"
        if self.graph_filter == "dreams":
            # Non-reality evolved space: dreams + promoted concepts
            return tier_value in ("dream", "concept")
        return True

    def cycle_graph_filter(self):
        """Cycle combined → reality → dreams."""
        idx = GRAPH_FILTERS.index(self.graph_filter) if self.graph_filter in GRAPH_FILTERS else 0
        self.graph_filter = GRAPH_FILTERS[(idx + 1) % len(GRAPH_FILTERS)]
        self.status_msg = f"Graph filter: {self.graph_filter.upper()}..."
        self.mark_graph_dirty()
        self.maybe_update_graph_layout(force=True)
        self.status_msg = f"Graph filter: {self.graph_filter.upper()} ({len(self.graph_nodes)} nodes)"

    def _pil_to_surface(self, pil_image, size):
        """Cache pygame surfaces — PIL→bytes conversion is expensive at 30fps."""
        if pil_image is None:
            return None
        try:
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            key = (id(pil_image), pil_image.size, size)
            if key == self._cached_dream_key and self._cached_dream_surface is not None:
                return self._cached_dream_surface
            data = pil_image.tobytes()
            surface = pygame.image.frombytes(data, pil_image.size, "RGB")
            if size != pil_image.size:
                surface = pygame.transform.smoothscale(surface, size)
            self._cached_dream_key = key
            self._cached_dream_surface = surface
            return surface
        except Exception as e:
            print(f"[UI] Image surface conversion failed: {e}")
            return None

    def _snapshot_world_state(self):
        """Grab UI state under lock briefly — pygame drawing stays outside the lock."""
        acquired = self.world.lock.acquire(blocking=False)
        if not acquired:
            if self._last_snap is not None:
                return self._last_snap
            return self._empty_snap

        try:
            snap = {
                "age": self.world.age,
                "active_concepts": set(self.world.active_concepts),
                "current_image": self.world.current_image,
                "total_living": len(self.world.memory.get_living_ids()),
                "total_ingested": self.world.memory.count_ingested_files(),
            }
            self._last_snap = snap
            return snap
        finally:
            self.world.lock.release()

    def mark_graph_dirty(self):
        self._graph_dirty = True

    def maybe_update_graph_layout(self, force=False):
        if self.view_mode != "graph":
            return
        if not force and not self._graph_dirty:
            return
        with self.world.lock:
            age = self.world.age
        if not force and (age - self._graph_layout_age) < self._graph_layout_interval:
            return
        self.update_graph_layout()
        self._graph_layout_age = age
        self._graph_dirty = False

    def update_graph_layout(self):
        """Builds a NetworkX graph from living concepts in the database."""
        with self.world.lock:
            embeddings = []
            node_ids = []
            tiers = {}
            pending_edges = []

            # Fetch all living concepts from database
            cursor = self.world.memory.conn.cursor()
            cursor.execute("SELECT id, embedding, parent_ids, generation FROM concepts")
            
            for row in cursor.fetchall():
                c_id = row[0]
                emb_json = row[1]
                parent_ids_json = row[2]
                generation = row[3]
                
                if not emb_json:
                    continue
                    
                import json
                try:
                    emb = np.array(json.loads(emb_json), dtype=np.float32).flatten()
                except Exception:
                    continue
                    
                if emb.size == 0:
                    continue

                try:
                    parent_ids = json.loads(parent_ids_json) if parent_ids_json else []
                except Exception:
                    parent_ids = []
                    
                for pid in parent_ids:
                    pending_edges.append((pid, c_id))

                embeddings.append(emb)
                node_ids.append(c_id)
                
                # Determine tier based on generation
                if generation == 0:
                    tiers[c_id] = "reality"
                elif generation < 100:
                    tiers[c_id] = "concept"
                else:
                    tiers[c_id] = "dream"

            visible = set(node_ids)
            edges = [(a, b) for a, b in pending_edges if a in visible and b in visible]

        self.graph_nodes = list(node_ids)
        self.graph_edges = edges
        self.graph_tiers = tiers

        if len(embeddings) > 1:
            try:
                pca = PCA(n_components=2)
                emb_stack = np.stack(embeddings)
                n_fit = min(len(embeddings), self._graph_max_pca_fit)
                if n_fit < len(embeddings):
                    fit_idx = np.random.choice(len(embeddings), n_fit, replace=False)
                    pca.fit(emb_stack[fit_idx])
                    coords = pca.transform(emb_stack)
                else:
                    coords = pca.fit_transform(emb_stack)
                max_val = np.max(np.abs(coords))
                if max_val > 0:
                    coords = coords / max_val
                self.graph_pos = {node_ids[i]: coords[i] for i in range(len(node_ids))}
            except Exception as e:
                print(f"PCA Layout failed: {e}")
                if len(node_ids) <= 800:
                    G2 = nx.DiGraph()
                    G2.add_nodes_from(node_ids)
                    G2.add_edges_from(edges)
                    self.graph_pos = nx.spring_layout(G2, seed=42)
                else:
                    # spring_layout on 7k nodes freezes the UI — grid fallback
                    side = int(np.ceil(np.sqrt(len(node_ids))))
                    self.graph_pos = {
                        node_ids[i]: np.array([
                            (i % side) / max(side - 1, 1) * 2 - 1,
                            (i // side) / max(side - 1, 1) * 2 - 1,
                        ])
                        for i in range(len(node_ids))
                    }
        elif len(embeddings) == 1:
            self.graph_pos = {node_ids[0]: np.array([0.0, 0.0])}
        else:
            self.graph_pos = {}

    def draw_graph(self, snap):
        """Draws the concept graph on the right panel, colored by memory tier."""
        cx, cy = 736, 310
        scale = 220
        active = snap.get("active_concepts", set())

        for edge in self.graph_edges:
            if edge[0] in self.graph_pos and edge[1] in self.graph_pos:
                p1 = self.graph_pos[edge[0]]
                p2 = self.graph_pos[edge[1]]
                x1, y1 = int(p1[0] * scale + cx), int(p1[1] * scale + cy)
                x2, y2 = int(p2[0] * scale + cx), int(p2[1] * scale + cy)
                pygame.draw.line(self.screen, (55, 55, 65), (x1, y1), (x2, y2), 1)

        for node in self.graph_nodes:
            if node not in self.graph_pos:
                continue
            p = self.graph_pos[node]
            x, y = int(p[0] * scale + cx), int(p[1] * scale + cy)

            tier = self.graph_tiers.get(node, "dream")
            color = TIER_COLORS.get(tier, BLUE)
            radius = 3 if tier == "reality" else (5 if tier == "concept" else 4)
            if node in active:
                color = WHITE
                radius = 7
                pygame.draw.circle(self.screen, RED, (x, y), radius + 2, 1)

            pygame.draw.circle(self.screen, color, (x, y), radius)

        self.draw_text(f"Layer: {self.graph_filter.upper()}", 490, 88, WHITE)
        self.draw_text(f"n={len(self.graph_nodes)}", 490, 108, GRAY)
        self.draw_text("Reality", 490, 545, GREEN)
        self.draw_text("Concept", 580, 545, BLUE)
        self.draw_text("Dream", 690, 545, ORANGE)
        self.draw_text("[T] cycle layer", 790, 545, GRAY)

        thumb = self._pil_to_surface(snap.get("current_image"), (96, 96))
        if thumb is not None:
            pygame.draw.rect(self.screen, GRAY, (892, 488, 96, 96), 1)
            self.screen.blit(thumb, (892, 488))

    def draw_ui(self):
        snap = self._snapshot_world_state()
        self.screen.fill(BLACK)

        self.draw_text("VISUAL UNIVERSE LAB (PURE PIXEL EVOLUTION)", 20, 20, BLUE, self.title_font)
        self.draw_text(f"Generation: {snap['age']}", 20, 80)
        self.draw_text(f"Living Concepts: {snap.get('total_living', 0)}", 20, 110)
        self.draw_text(f"Real Images Ingested: {snap.get('total_ingested', 0)}", 20, 140, GREEN)
        
        loss = self.trainer.get_recent_loss()
        self.draw_text(f"Model Loss (MAE): {loss:.4f}", 20, 170, ORANGE)

        if self.last_metrics:
            mode = self.last_metrics.get('mode', 'Recombination')
            self.draw_text(f"Last Evolution Step ({mode}):", 20, 200, BLUE)
            
            if "error" in self.last_metrics:
                self.draw_text(f"Error: {self.last_metrics['error']}", 20, 230, RED)
            else:
                self.draw_text(f"Child ID: {self.last_metrics.get('child_id')}", 20, 230)
                self.draw_text(f"Parents: {self.last_metrics.get('parents')}", 20, 260)
                
                coherence = self.last_metrics.get('coherence', 0)
                novelty = self.last_metrics.get('novelty', 0)
                
                self.draw_text(f"Aesthetics: {coherence:.3f}", 20, 290, GREEN if coherence > 0.6 else GRAY)
                self.draw_text(f"Novelty: {novelty:.3f}", 20, 320, ORANGE if novelty > 0.6 else GRAY)
                
                tags = self.last_metrics.get('tags', [])
                if tags:
                    self.draw_text(f"Tags: {', '.join(tags)}", 20, 350, BLUE)

        self.draw_text("Controls:", 250, 680, GRAY)
        self.draw_text("[SPACE] Step Evolution", 250, 710)
        self.draw_text("[A] Toggle Auto-Evolve", 250, 740)
        self.draw_text("[I] Ingest Dataset / Cancel", 250, 770)

        self.draw_text(f"Auto-Evolve: {'ON' if self.auto_evolve else 'OFF'}", 20, 840,
                       GREEN if self.auto_evolve else GRAY)
        self.draw_text(
            f"Evolve: {self.evolve_accepted}/{self.evolve_attempts} accepted"
            + (" (running...)" if self._evolve_busy else ""),
            20, 810,
            ORANGE if self._evolve_busy else GRAY,
        )
        ingest_label = "PAUSED (auto)" if self.auto_evolve else ("ON" if self.is_ingesting else "OFF")
        ingest_color = ORANGE if self.auto_evolve else (GREEN if self.is_ingesting else GRAY)
        self.draw_text(f"Reality Ingest: {ingest_label}", 20, 870, ingest_color)
        
        self.draw_text(f"Status: {self.status_msg}", 20, 950, BLUE)

        pygame.draw.rect(self.screen, (30, 30, 35), (480, 80, 512, 512))

        if self.view_mode == "dream":
            dream_surface = self._pil_to_surface(snap.get("current_image"), (512, 512))
            if dream_surface is not None:
                self.screen.blit(dream_surface, (480, 80))
        elif self.view_mode == "graph":
            self.draw_graph(snap)

        self.draw_timeline(snap["age"])
        pygame.display.flip()
            
    def load_timeline_snapshot(self):
        """Loads a historical image at self.timeline_age."""
        with self.world.lock:
            cursor = self.world.memory.conn.cursor()
            cursor.execute(
                "SELECT id, image_path FROM concepts ORDER BY ABS(generation - ?) LIMIT 1",
                (self.timeline_age,),
            )
            res = cursor.fetchone()
            if not res:
                return
            concept_id, image_path = res

        try:
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path).convert("RGB")
                with self.world.lock:
                    self.world.current_image = img
        except Exception as e:
            print(f"[Timeline] Failed to load gen~{self.timeline_age} (id={concept_id}): {e}")

    def draw_timeline(self, age):
        """Draw a timeline slider at the bottom of the right panel."""
        cx = 480
        cy = 620
        width = 512
        height = 40
        pygame.draw.rect(self.screen, (40, 40, 45), (cx, cy, width, height))

        max_age = max(1, age)
        
        # Draw line
        pygame.draw.line(self.screen, GRAY, (cx + 20, cy + 20), (cx + width - 20, cy + 20), 2)
        
        # Draw current age marker
        handle_x = cx + 20 + int((self.timeline_age / max_age) * (width - 40))
        pygame.draw.circle(self.screen, BLUE, (handle_x, cy + 20), 6)
        
        self.draw_text(f"Timeline Scrub: Gen {self.timeline_age}", cx + 20, cy + 45, BLUE)

    def evolve_step(self):
        if not self._evolve_lock.acquire(blocking=False):
            return
        self._evolve_busy = True
        self.evolve_attempts += 1
        self.status_msg = f"Dreaming... (attempt {self.evolve_attempts})"
        try:
            success, metrics = self.world.step()
        except Exception as e:
            print(f"[Evolve] step crashed (kept alive): {e}")
            import traceback
            traceback.print_exc()
            self.status_msg = f"Evolve error: {e}"
            return
        finally:
            self._evolve_busy = False
            self._evolve_lock.release()

        self.last_metrics = metrics
        if success:
            self.evolve_accepted += 1
            mode = metrics.get("mode", "?")
            self.status_msg = f"Accepted ({mode}) — {self.evolve_accepted}/{self.evolve_attempts}"
            if not self.scrubbing_timeline and self.world.age > self.timeline_age:
                self.timeline_age = self.world.age
            if self.view_mode == "graph":
                self.mark_graph_dirty()
                self.maybe_update_graph_layout()
        else:
            mode = metrics.get("mode", "?")
            reason = metrics.get("rejection_reason", "unknown")
            self.status_msg = (
                f"Rejected ({mode}: {reason}) — {self.evolve_accepted}/{self.evolve_attempts}"
            )

    def _auto_evolve_loop(self):
        """Continuous evolution while auto is on — one step at a time, no key-repeat issues."""
        while self.auto_evolve and self.running:
            self.evolve_step()
            if self.auto_evolve and self.running:
                time.sleep(0.05)

    def _start_auto_evolve(self):
        if self._auto_evolve_thread is not None and self._auto_evolve_thread.is_alive():
            return
        self._auto_evolve_thread = threading.Thread(
            target=self._auto_evolve_loop, daemon=True, name="AutoEvolve"
        )
        self._auto_evolve_thread.start()

    def _start_manual_evolve(self):
        if self._evolve_busy:
            return
        if self._manual_evolve_thread is not None and self._manual_evolve_thread.is_alive():
            return
        self._manual_evolve_thread = threading.Thread(
            target=self.evolve_step, daemon=True, name="ManualEvolve"
        )
        self._manual_evolve_thread.start()

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
            self.mark_graph_dirty()
            self.maybe_update_graph_layout(force=True)

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

    def run(self):
        clock = pygame.time.Clock()
        
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
                    if getattr(event, "repeat", False):
                        continue
                    if event.key == pygame.K_SPACE:
                        self._start_manual_evolve()
                    elif event.key == pygame.K_a:
                        self.auto_evolve = not self.auto_evolve
                        if self.auto_evolve:
                            self.cancel_ingest = True
                            self.status_msg = "Auto-Evolve ON — reality ingest paused"
                            self._start_auto_evolve()
                        else:
                            self.status_msg = "Auto-Evolve OFF — reality ingest resumed"
                    elif event.key == pygame.K_i:
                        if self.auto_evolve:
                            self.status_msg = "Reality ingest paused while Auto-Evolve is ON"
                        elif self.is_ingesting:
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
                            self.mark_graph_dirty()
                            self.maybe_update_graph_layout(force=True)
                            self.status_msg = f"Graph view ({self.graph_filter}) — [T] cycles layers"
                        else:
                            self.view_mode = "dream"
                            self.status_msg = "Dream view active."
                    elif event.key == pygame.K_t:
                        if self.view_mode == "graph":
                            self.cycle_graph_filter()
                        else:
                            self.status_msg = "Switch to graph view [V] first, then [T] filters layers"
                    elif event.key == pygame.K_m:
                        if hasattr(self.world, 'mode_library'):
                            modes = [None, 'crystalline', 'fluid', 'organic', 'geometric', 'chaotic', 'smooth']
                            current_idx = modes.index(self.world.mode_library.active_mode_preference)
                            next_idx = (current_idx + 1) % len(modes)
                            self.world.mode_library.set_mode_preference(modes[next_idx])
                            self.status_msg = f"Mode: {modes[next_idx] or 'AUTO'}"
                        else:
                            self.status_msg = "Mode library not available."

            # Background slow ingestion — paused while auto-evolve runs
            if not self.auto_evolve:
                background_ingest_timer += 1
                if background_ingest_timer >= 60 and not self.is_ingesting:
                    background_ingest_timer = 0
                    if ingest_thread is None or not ingest_thread.is_alive():
                        def bg_task():
                            if self.auto_evolve:
                                return
                            # Get a random uningested file from dataset dir
                            import os
                            import random
                            dataset_dir = "dataset"
                            if os.path.exists(dataset_dir):
                                valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
                                files = [f for f in os.listdir(dataset_dir) if os.path.splitext(f)[1].lower() in valid_exts]
                                random.shuffle(files)
                                for filename in files:
                                    with self.world.lock:
                                        if self.world.memory.is_file_ingested(filename):
                                            continue
                                    filepath = os.path.join(dataset_dir, filename)
                                    success, _ = self.world.ingest_image(filepath)
                                    if success:
                                        self.mark_graph_dirty()
                                        break
                                        
                        ingest_thread = threading.Thread(target=bg_task, daemon=True)
                        ingest_thread.start()
                    
            self._frame += 1
            if self.view_mode == "graph":
                self.maybe_update_graph_layout()
            self.draw_ui()
            clock.tick(30)
            
        self.trainer.stop()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = UniverseApp()
    app.run()
