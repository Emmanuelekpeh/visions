"""
Sparse Image Logger
Saves interesting concept images without flooding storage.
"""
import os
from datetime import datetime
from PIL import Image
from typing import Optional

class ImageLogger:
    """
    Saves concept images sparsely based on importance criteria.
    """
    
    def __init__(self, base_dir: str = "outputs", save_interval: int = 25):
        """
        Args:
            base_dir: Root directory for all outputs
            save_interval: Save every N generations
        """
        self.base_dir = base_dir
        self.save_interval = save_interval
        
        # Create session folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, f"run_{timestamp}")
        
        # Create subdirectories
        self.dirs = {
            "milestones": os.path.join(self.session_dir, "milestones"),
            "promotions": os.path.join(self.session_dir, "promotions"),
            "high_fitness": os.path.join(self.session_dir, "high_fitness"),
            "combinations": os.path.join(self.session_dir, "combinations")
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        print(f"[ImageLogger] Session: {self.session_dir}")
        
    def should_save_milestone(self, generation: int) -> bool:
        """Save every N generations."""
        return generation % self.save_interval == 0
        
    def save_milestone(self, image: Image.Image, generation: int, 
                      concept_id: int, metadata: dict):
        """Save a milestone image (every N generations)."""
        filename = f"gen{generation:05d}_id{concept_id:05d}.png"
        filepath = os.path.join(self.dirs["milestones"], filename)
        
        image.save(filepath)
        
        # Save metadata alongside
        meta_path = filepath.replace(".png", ".txt")
        with open(meta_path, 'w') as f:
            f.write(f"Generation: {generation}\n")
            f.write(f"Concept ID: {concept_id}\n")
            f.write(f"Fitness: {metadata.get('fitness', 0):.3f}\n")
            f.write(f"Confidence: {metadata.get('confidence', 0):.3f}\n")
            f.write(f"Tier: {metadata.get('tier', 'unknown')}\n")
            f.write(f"Mode: {metadata.get('mode', 'unknown')}\n")
            
    def save_promotion(self, image: Image.Image, generation: int,
                      concept_id: int, from_tier: str, to_tier: str):
        """Save when a concept gets promoted."""
        filename = f"promoted_gen{generation:05d}_id{concept_id:05d}_{from_tier}_to_{to_tier}.png"
        filepath = os.path.join(self.dirs["promotions"], filename)
        image.save(filepath)
        print(f"[ImageLogger] Saved promotion: {filename}")
        
    def save_high_fitness(self, image: Image.Image, generation: int,
                         concept_id: int, fitness: float):
        """Save concepts with exceptional fitness."""
        if fitness > 0.75:  # Only save really good ones
            filename = f"fitness{fitness:.3f}_gen{generation:05d}_id{concept_id:05d}.png"
            filepath = os.path.join(self.dirs["high_fitness"], filename)
            image.save(filepath)
            
    def save_combination(self, image: Image.Image, generation: int,
                        concept_id: int, parent_a: int, parent_b: int, 
                        success: float):
        """Save successful recombinations."""
        if success > 0.70:  # Only save successful combinations
            filename = f"combo_gen{generation:05d}_id{concept_id:05d}_p{parent_a}_{parent_b}.png"
            filepath = os.path.join(self.dirs["combinations"], filename)
            image.save(filepath)
            
    def get_stats(self) -> dict:
        """Get counts of saved images."""
        stats = {}
        for name, path in self.dirs.items():
            if os.path.exists(path):
                stats[name] = len([f for f in os.listdir(path) if f.endswith('.png')])
            else:
                stats[name] = 0
        return stats
