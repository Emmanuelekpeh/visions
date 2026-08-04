import os
import time
import random
import threading
import torch
import numpy as np
from PIL import Image

from database import WorldMemory
from vision import VisionSystem
from engines import HybridMAE, load_image, save_image, apply_posterize, apply_halftone, apply_dither

class WorldState:
    """
    The core evolutionary loop.
    Uses pure pixel-space evolution via the Hybrid MAE model.
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.lock = threading.RLock()
        
        # Core systems
        self.memory = WorldMemory()
        self.vision = VisionSystem(device=device)
        self.model = HybridMAE().to(device)
        
        # State
        self.age = self.memory.get_max_generation()
        self.current_image = None
        self.active_concepts = set()
        
        # Load model weights if they exist
        self.model_weights_path = "hybrid_mae_weights.pt"
        if os.path.exists(self.model_weights_path):
            try:
                self.model.load_state_dict(torch.load(self.model_weights_path, map_location=self.device, weights_only=True))
                print(f"Loaded existing model weights from {self.model_weights_path}")
            except Exception as e:
                print(f"Failed to load model weights: {e}")
        
        # Ensure output directory exists
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join("outputs", "evolved", f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        print(f"World initialized at Generation {self.age}. Saving to {self.run_dir}")

    def ingest_image(self, filepath: str):
        """Ingest a real image into the database as a reality anchor."""
        with self.lock:
            if self.memory.is_file_ingested(filepath):
                return False, None
                
            try:
                img = Image.open(filepath).convert("RGB")
                img = img.resize((512, 512), Image.BILINEAR)
                
                # Extract features
                embedding = self.vision.extract_concept(img).cpu().numpy()[0]
                tags = self.vision.generate_tags(torch.from_numpy(embedding).to(self.device))
                aesthetics = self.vision.score_aesthetics(torch.from_numpy(embedding).to(self.device))
                
                # Add to DB using the original filepath (no need to duplicate)
                c_id = self.memory.add_concept(
                    embedding=embedding,
                    image_path=filepath,
                    parent_ids=[],
                    generation=0,
                    coherence=aesthetics.get("overall", 0.5),
                    novelty=1.0,  # Real images are maximally novel
                    tags=tags,
                    generator_version=-1
                )
                
                self.memory.mark_file_ingested(filepath, c_id)
                self.current_image = img
                self.active_concepts = {c_id}
                
                return True, c_id
            except Exception as e:
                print(f"Failed to ingest {filepath}: {e}")
                return False, None

    def _get_fitness(self, concept_id: int) -> float:
        """Calculate fitness for selection: Aesthetics + Novelty."""
        concept = self.memory.get_concept(concept_id)
        if not concept:
            return 0.0
        
        # concept tuple: (id, embedding, image_path, parent_ids, generation, coherence, novelty, ...)
        coherence = concept[5] or 0.5
        novelty = concept[6] or 0.5
        
        return (coherence * 0.6) + (novelty * 0.4)

    def _tournament_selection(self, k=3) -> int:
        """Select a parent using tournament selection."""
        living_ids = list(self.memory.get_living_ids())
        if not living_ids:
            return None
            
        candidates = random.sample(living_ids, min(k, len(living_ids)))
        best_id = None
        best_fitness = -1.0
        
        for cid in candidates:
            fitness = self._get_fitness(cid)
            if fitness > best_fitness:
                best_fitness = fitness
                best_id = cid
                
        return best_id

    def step(self):
        """
        One step of evolution:
        1. Select parents (or use Genesis mode if < 2 concepts exist)
        2. Recombine using Hybrid MAE (or generate from noise)
        3. Evaluate
        4. Save
        """
        with self.lock:
            living_ids = self.memory.get_living_ids()
            
            is_genesis = len(living_ids) < 2
            
            if is_genesis:
                # Genesis Mode: Generate from a blank canvas
                parent_a_id = -1
                parent_b_id = -1
                
                # We don't need parent images for a blank canvas generation
                img_a, img_b = None, None
            else:
                # 1. Selection
                parent_a_id = self._tournament_selection()
                parent_b_id = self._tournament_selection()
                
                if parent_a_id is None or parent_b_id is None:
                    return False, {"error": "Selection failed"}
                    
                path_a = self.memory.get_image_path_for_concept(parent_a_id)
                path_b = self.memory.get_image_path_for_concept(parent_b_id)
                
                if not path_a or not path_b or not os.path.exists(path_a) or not os.path.exists(path_b):
                    return False, {"error": "Parent images missing"}
                    
                img_a = load_image(path_a).to(self.device)
                img_b = load_image(path_b).to(self.device)
                
            # 2. Recombination / Generation
            try:
                if is_genesis:
                    # Generate from a blank canvas using learned mask tokens
                    child_tensor = self.model.generate_from_blank()
                else:
                    # Recombine using the model
                    child_tensor = self.model.recombine(img_a, img_b, mask_ratio=0.5)
                
                # Convert back to PIL for vision system
                img_np = child_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                child_img = Image.fromarray(img_np)
                
                # --- Stylistic Mutations (40% chance total) ---
                style_tag = None
                r = random.random()
                if r < 0.15:
                    child_img = apply_posterize(child_img, bits=3)
                    style_tag = "style: posterized"
                elif r < 0.30:
                    child_img = apply_halftone(child_img, sample=6)
                    style_tag = "style: halftone"
                elif r < 0.40:
                    child_img = apply_dither(child_img, colors=16)
                    style_tag = "style: dithered"
                
            except Exception as e:
                print(f"Recombination failed: {e}")
                return False, {"error": str(e)}
                
            # 3. Evaluation
            self.age += 1
            embedding = self.vision.extract_concept(child_img).cpu().numpy()[0]
            tags = self.vision.generate_tags(torch.from_numpy(embedding).to(self.device))
            
            if style_tag:
                tags.append(style_tag)
                
            aesthetics = self.vision.score_aesthetics(torch.from_numpy(embedding).to(self.device))
            
            # Calculate Novelty (distance to nearest neighbors)
            neighbors = self.memory.get_nearest_concepts(embedding, k=5)
            novelty = 1.0
            if neighbors:
                avg_dist = sum(dist for _, dist in neighbors) / len(neighbors)
                novelty = min(1.0, avg_dist / 2.0) # Normalize
                
            coherence = aesthetics.get("overall", 0.5)
            
            # 4. Storage
            out_path = os.path.join(self.run_dir, f"gen_{self.age}.png")
            child_img.save(out_path)
            
            # Use empty list for genesis parents so it doesn't break DB foreign keys or graph logic
            stored_parents = [] if is_genesis else [parent_a_id, parent_b_id]
            
            c_id = self.memory.add_concept(
                embedding=embedding,
                image_path=out_path,
                parent_ids=stored_parents,
                generation=self.age,
                coherence=coherence,
                novelty=novelty,
                tags=tags,
                generator_version=1
            )
            
            self.current_image = child_img
            self.active_concepts = {c_id}
            
            # 5. Culling (Keep population size manageable)
            if len(living_ids) > 1000:
                self._cull_population(100)
                
            return True, {
                "age": self.age,
                "parents": stored_parents,
                "child_id": c_id,
                "coherence": coherence,
                "novelty": novelty,
                "tags": tags,
                "mode": "Genesis (Blank Canvas)" if is_genesis else "Recombination"
            }
            
    def _cull_population(self, num_to_remove: int):
        """Remove the lowest fitness concepts."""
        living_ids = list(self.memory.get_living_ids())
        # Don't cull real images
        real_ids = set(self.memory.get_real_image_anchor_ids())
        cullable = [cid for cid in living_ids if cid not in real_ids]
        
        if not cullable:
            return
            
        # Sort by fitness ascending
        cullable.sort(key=lambda cid: self._get_fitness(cid))
        
        for cid in cullable[:num_to_remove]:
            self.memory.prune_concept(cid, reason="Low Fitness Culling")
