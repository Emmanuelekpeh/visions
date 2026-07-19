"""
Epistemic Memory System
Confidence-based concept validation with prediction error tracking.

Core insight: Real ≠ Good. Derived ≠ Bad.
What matters: How confident are we this concept deserves to exist?
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time

class MemoryTier(Enum):
    """Memory ecosystem tiers."""
    LANDMARK = "landmark"      # Immutable real images (border checkpoints)
    WORKING = "working"        # Proven evolved concepts
    FRONTIER = "frontier"      # Experimental, unproven concepts

@dataclass
class ConceptConfidence:
    """Epistemic confidence tracking for a concept."""
    concept_id: int
    embedding: np.ndarray      # Concept embedding (for reality distance computation)
    confidence: float          # 0-1: How certain we are this concept is valuable
    reality_distance: float    # Distance to nearest real image
    prediction_errors: List[float]  # History of prediction errors
    validation_count: int      # How many times validated
    tier: MemoryTier          # Which memory ecosystem
    
    # Metadata
    creation_time: float
    last_validation: float
    promotion_eligible: bool
    
    def mean_prediction_error(self) -> float:
        """Average prediction error (learning signal)."""
        if not self.prediction_errors:
            return 0.0
        return float(np.mean(self.prediction_errors[-10:]))  # Last 10
        
    def is_stable(self) -> bool:
        """Has low prediction error variance (trustworthy)."""
        if len(self.prediction_errors) < 5:
            return False
        return float(np.std(self.prediction_errors[-10:])) < 0.15

    def get_mutation_energy(self, current_time: float) -> float:
        """
        Thermodynamic constraint: A concept's budget for transformation.
        mutation_energy = confidence * age * stability * unused_potential
        """
        # Age factor (older concepts have more stored energy, but caps out)
        age_generations = max(1, current_time - self.creation_time)
        age_factor = min(2.0, np.log10(age_generations + 10) / 2.0)
        
        # Stability factor (stable concepts can afford bigger leaps)
        stability_factor = 1.2 if self.is_stable() else 0.8
        
        # Unused potential (concepts that haven't mutated recently build up pressure)
        time_since_validation = max(0, current_time - self.last_validation)
        unused_potential = min(1.5, 1.0 + (time_since_validation / 100.0))
        
        # Base energy is confidence
        energy = self.confidence * age_factor * stability_factor * unused_potential
        return float(max(0.1, energy))


class EpistemicMemory:
    """
    Confidence-based memory with three-tier ecosystem and prediction error tracking.
    """
    
    def __init__(self, vision_system, faiss_index):
        self.vision = vision_system
        self.faiss_index = faiss_index
        
        # Three-tier memory
        self.landmark_concepts: Dict[int, ConceptConfidence] = {}
        self.working_concepts: Dict[int, ConceptConfidence] = {}
        self.frontier_concepts: Dict[int, ConceptConfidence] = {}
        
        # Prediction tracking
        self.expected_scores: Dict[int, float] = {}
        
        # Promotion thresholds (RELAXED to enable progression)
        self.frontier_survival_generations = 3  # Reduced from 5
        self.promotion_confidence_threshold = 0.50  # Reduced from 0.65
        
    def add_real_image(self, concept_id: int, embedding: np.ndarray) -> ConceptConfidence:
        """Add a real image to landmark memory (immutable, max confidence)."""
        conf = ConceptConfidence(
            concept_id=concept_id,
            embedding=embedding,
            confidence=1.0,  # Real images start with full confidence
            reality_distance=0.0,  # They ARE reality
            prediction_errors=[],
            validation_count=0,
            tier=MemoryTier.LANDMARK,
            creation_time=time.time(),
            last_validation=time.time(),
            promotion_eligible=False
        )
        self.landmark_concepts[concept_id] = conf
        return conf
        
    def add_derived_concept(self, concept_id: int, embedding: np.ndarray, 
                           parent_confidences: List[float]) -> ConceptConfidence:
        """
        Add a derived concept to frontier (unproven).
        Confidence inherited from parents.
        """
        # Inherit confidence from parents
        if parent_confidences:
            inherited_confidence = 0.9 * np.mean(parent_confidences)  # Slight decay
        else:
            inherited_confidence = 0.5  # Orphan concepts start uncertain
            
        # Compute reality distance
        reality_dist = self._compute_reality_distance(embedding)
        
        # Start in frontier (must prove itself)
        conf = ConceptConfidence(
            concept_id=concept_id,
            embedding=embedding,
            confidence=inherited_confidence,
            reality_distance=reality_dist,
            prediction_errors=[],
            validation_count=0,
            tier=MemoryTier.FRONTIER,
            creation_time=time.time(),
            last_validation=time.time(),
            promotion_eligible=False
        )
        
        self.frontier_concepts[concept_id] = conf
        return conf
        
    def _compute_reality_distance(self, embedding: np.ndarray) -> float:
        """
        Compute distance to nearest REAL IMAGE.
        This is the "return-to-reality test".
        """
        # Get all landmark embeddings
        landmark_ids = list(self.landmark_concepts.keys())
        if not landmark_ids:
            return 999.0  # No reality anchors
            
        # Find nearest landmark via linear search
        # (In production, maintain separate FAISS index for landmarks)
        min_dist = float('inf')
        
        for lid in landmark_ids:
            landmark_conf = self.landmark_concepts[lid]
            landmark_emb = landmark_conf.embedding
            
            # L2 distance
            dist = np.linalg.norm(embedding - landmark_emb)
            min_dist = min(min_dist, dist)
            
        return float(min_dist)
        
    def validate_concept(self, concept_id: int, latent: torch.Tensor,
                        expected_score: float, actual_score: float) -> float:
        """
        Core validation: Decode → Re-encode → Check reality distance.
        Track prediction error as learning signal.
        
        Returns: Updated confidence
        """
        # Find concept in appropriate tier
        conf = self._get_concept(concept_id)
        if conf is None:
            return 0.0
            
        # 1. Prediction Error (surprise = learning)
        prediction_error = abs(actual_score - expected_score)
        conf.prediction_errors.append(prediction_error)
        
        # 2. Round-trip validation (decode → re-encode)
        try:
            # Decode latent to image
            image = self.vision.decode_latent_to_image(latent)
            
            # Re-encode
            new_embedding = self.vision.extract_concept(image)
            new_embedding_np = new_embedding.cpu().numpy().flatten()
            
            # Compute reality distance on re-encoded embedding
            reality_dist = self._compute_reality_distance(new_embedding_np)
            conf.reality_distance = reality_dist
            
        except Exception as e:
            print(f"Validation failed for concept {concept_id}: {e}")
            # Penalize concepts that can't be decoded
            conf.confidence *= 0.8
            return conf.confidence
            
        # 3. Update confidence based on multiple signals
        validation_bonus = 0.0
        
        # Moderate prediction error = learning (good!)
        if 0.1 < prediction_error < 0.4:
            validation_bonus += 0.05  # Sweet spot for learning
        elif prediction_error > 0.6:
            validation_bonus -= 0.05  # Too unpredictable
            
        # Close to reality = good
        if reality_dist < 0.3:
            validation_bonus += 0.05
        elif reality_dist > 1.0:
            validation_bonus -= 0.1  # Drifted too far
            
        # Stable predictions = trustworthy
        if conf.is_stable():
            validation_bonus += 0.03
            
        # Update confidence
        conf.confidence = np.clip(conf.confidence + validation_bonus, 0.0, 1.0)
        conf.validation_count += 1
        conf.last_validation = time.time()
        
        # Check promotion eligibility
        if conf.tier == MemoryTier.FRONTIER:
            generations_survived = conf.validation_count
            if generations_survived >= self.frontier_survival_generations:
                conf.promotion_eligible = True
                
        return conf.confidence
        
    def promote_frontier_concepts(self):
        """Promote frontier concepts that proved themselves to working memory."""
        promoted = []
        
        for c_id, conf in list(self.frontier_concepts.items()):
            if (conf.promotion_eligible and 
                conf.confidence > self.promotion_confidence_threshold and
                conf.reality_distance < 1.2):  # Increased from 0.8 to allow more distant concepts
                
                # Promote to working memory
                conf.tier = MemoryTier.WORKING
                self.working_concepts[c_id] = conf
                del self.frontier_concepts[c_id]
                promoted.append(c_id)
                
        return promoted
        
    def quarantine_dangerous_concepts(self):
        """
        Quarantine frontier concepts that drifted too far or have low confidence.
        """
        quarantined = []
        
        for c_id, conf in list(self.frontier_concepts.items()):
            # Danger signals (RELAXED to prevent over-quarantine)
            too_far = conf.reality_distance > 2.0  # Increased from 1.5
            too_uncertain = conf.confidence < 0.20  # Reduced from 0.25
            unstable = len(conf.prediction_errors) > 10 and not conf.is_stable()  # Increased from 5
            
            if too_far or too_uncertain or unstable:
                # Remove from frontier
                del self.frontier_concepts[c_id]
                quarantined.append(c_id)
                
        return quarantined
        
    def _get_concept(self, concept_id: int) -> Optional[ConceptConfidence]:
        """Find concept in any tier."""
        if concept_id in self.landmark_concepts:
            return self.landmark_concepts[concept_id]
        elif concept_id in self.working_concepts:
            return self.working_concepts[concept_id]
        elif concept_id in self.frontier_concepts:
            return self.frontier_concepts[concept_id]
        return None
        
    def get_selection_weights(self, concept_ids: List[int]) -> np.ndarray:
        """
        Get selection weights for concepts based on confidence.
        Replaces naive reality boost.
        """
        weights = []
        for c_id in concept_ids:
            conf = self._get_concept(c_id)
            if conf is None:
                weights.append(0.1)  # Unknown concept, low weight
            else:
                # Weight = confidence × tier multiplier
                tier_multiplier = {
                    MemoryTier.LANDMARK: 1.5,  # Prefer proven anchors
                    MemoryTier.WORKING: 1.0,   # Equal weight
                    MemoryTier.FRONTIER: 0.5   # Risky experiments
                }[conf.tier]
                
                weights.append(conf.confidence * tier_multiplier)
                
        return np.array(weights)
        
    def get_status(self) -> Dict:
        """Get memory status."""
        return {
            "landmark_count": len(self.landmark_concepts),
            "working_count": len(self.working_concepts),
            "frontier_count": len(self.frontier_concepts),
            "avg_confidence_working": np.mean([c.confidence for c in self.working_concepts.values()]) if self.working_concepts else 0.0,
            "avg_confidence_frontier": np.mean([c.confidence for c in self.frontier_concepts.values()]) if self.frontier_concepts else 0.0,
            "avg_reality_distance": np.mean([c.reality_distance for c in list(self.working_concepts.values()) + list(self.frontier_concepts.values())]) if (self.working_concepts or self.frontier_concepts) else 0.0
        }
        
    def should_explore_frontier(self) -> bool:
        """
        Decide if we should try risky frontier concepts.
        Balance exploration vs exploitation.
        """
        total_concepts = len(self.working_concepts) + len(self.frontier_concepts)
        if total_concepts == 0:
            return True
            
        frontier_ratio = len(self.frontier_concepts) / total_concepts
        
        # If frontier is small, we can afford to explore
        # If frontier is large, we need to validate what we have
        return frontier_ratio < 0.3
        
    def save_to_database(self, db_conn):
        """Save epistemic memory state to database."""
        import sqlite3
        import json
        
        cursor = db_conn.cursor()
        
        print("Saving epistemic memory state...")
        saved_count = 0
        
        # Save all concepts (landmark, working, frontier)
        # Create snapshot to avoid RuntimeError during iteration
        all_concepts = {}
        all_concepts.update(self.landmark_concepts)
        all_concepts.update(self.working_concepts)
        all_concepts.update(self.frontier_concepts)
        
        concepts_snapshot = list(all_concepts.items())
        
        for concept_id, conf in concepts_snapshot:
            prediction_errors_json = json.dumps(conf.prediction_errors[-50:])  # Last 50
            
            cursor.execute('''
                INSERT OR REPLACE INTO epistemic_metadata
                (concept_id, tier, confidence, reality_distance, prediction_errors,
                 validation_count, creation_time, last_validation, promotion_eligible)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                concept_id,
                conf.tier.value,
                conf.confidence,
                conf.reality_distance,
                prediction_errors_json,
                conf.validation_count,
                conf.creation_time,
                conf.last_validation,
                1 if conf.promotion_eligible else 0
            ))
            saved_count += 1
        
        db_conn.commit()
        print(f"Saved epistemic state for {saved_count} concepts")
    
    def load_from_database(self, db_conn):
        """
        Load existing concepts from database into epistemic memory.
        
        Strategy:
        - Concepts from ingested_files → Landmark tier
        - Other concepts → Frontier tier (will be promoted based on performance)
        """
        import sqlite3
        import json
        
        cursor = db_conn.cursor()
        
        # Get real image IDs
        cursor.execute("SELECT concept_id FROM ingested_files")
        real_image_ids = {row[0] for row in cursor}
        
        # Load all concepts
        cursor.execute("SELECT id, embedding FROM concepts")
        
        landmark_count = 0
        frontier_count = 0
        
        for row in cursor:
            concept_id = row[0]
            embedding_json = row[1]
            
            try:
                import numpy as np
                embedding_np = np.array(json.loads(embedding_json), dtype=np.float32)
                
                if concept_id in real_image_ids:
                    # Real image → Landmark
                    self.add_real_image(concept_id, embedding_np)
                    landmark_count += 1
                else:
                    # Derived → Frontier (must prove itself)
                    self.add_derived_concept(concept_id, embedding_np, parent_confidences=[])
                    frontier_count += 1
                    
            except Exception as e:
                print(f"[WARN] Failed to load concept {concept_id}: {e}")
                continue
                
        # Try to load saved epistemic state if available
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='epistemic_metadata'")
        if cursor.fetchone():
            print("Loading saved epistemic state...")
            cursor.execute("SELECT * FROM epistemic_metadata")
            
            restored_count = 0
            for i, row in enumerate(cursor):
                if i % 500 == 0:
                    print(f"Processed {i} epistemic metadata rows...")
                concept_id = row[0]
                tier_str = row[1]
                confidence = row[2]
                reality_distance = row[3]
                prediction_errors_json = row[4]
                validation_count = row[5]
                creation_time = row[6]
                last_validation = row[7]
                promotion_eligible = bool(row[8])
                
                # Find concept in one of the tiers
                conf = None
                if concept_id in self.landmark_concepts:
                    conf = self.landmark_concepts[concept_id]
                elif concept_id in self.working_concepts:
                    conf = self.working_concepts[concept_id]
                elif concept_id in self.frontier_concepts:
                    conf = self.frontier_concepts[concept_id]
                
                if conf:
                    # Restore saved state
                    saved_tier = MemoryTier(tier_str)
                    conf.confidence = confidence
                    conf.reality_distance = reality_distance
                    conf.validation_count = validation_count
                    conf.creation_time = creation_time
                    conf.last_validation = last_validation
                    conf.promotion_eligible = promotion_eligible
                    
                    try:
                        conf.prediction_errors = json.loads(prediction_errors_json) if prediction_errors_json else []
                    except (json.JSONDecodeError, TypeError):
                        conf.prediction_errors = []
                    
                    # Move to correct tier if it changed
                    if saved_tier != conf.tier:
                        if saved_tier == MemoryTier.WORKING and conf.tier == MemoryTier.FRONTIER:
                            # Concept was promoted, move it
                            self.working_concepts[concept_id] = conf
                            del self.frontier_concepts[concept_id]
                            conf.tier = MemoryTier.WORKING
                            restored_count += 1
                    else:
                        restored_count += 1
            
            print(f"Restored epistemic state for {restored_count} concepts")
            
            # Recount after restoration
            landmark_count = len(self.landmark_concepts)
            working_count = len(self.working_concepts)
            frontier_count = len(self.frontier_concepts)
            print(f"Final state: {landmark_count} landmarks, {working_count} working, {frontier_count} frontier")
        else:
            print("No saved epistemic state found (first run)")
        
        return landmark_count, frontier_count


# Test
if __name__ == "__main__":
    print("Epistemic Memory System")
    print("=" * 60)
    print("\nCore Principles:")
    print("1. Confidence inheritance (not naive reality boost)")
    print("2. Return-to-reality validation (decode → re-encode)")
    print("3. Three-tier ecosystem (Landmark/Working/Frontier)")
    print("4. Prediction error tracking (surprise = learning)")
    print("5. Promotion system (prove yourself to advance)")
    print("\n[System ready for integration]")
