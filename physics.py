"""
Physics Engine
Defines the laws of nature for the universe.

Concepts are nouns. Transformations are verbs.
A transformation is not a law until it survives repeated experiments across changing conditions.

Hierarchy:
Event -> Pattern -> Hypothesis -> Species Law -> Universal Law
"""

import numpy as np
import json
import time
from enum import Enum
from typing import Dict, List, Set, Optional

class TransformationTier(Enum):
    EVENT = "event"                # Saw it happen once
    PATTERN = "pattern"            # Happened a few times
    HYPOTHESIS = "hypothesis"      # Testing it actively
    SPECIES_LAW = "species_law"    # Reliable in a specific domain
    UNIVERSAL_LAW = "universal_law" # Reliable everywhere
    FOSSILIZED_LAW = "fossilized_law" # Used to be a law, now forgotten

class FailureType(Enum):
    INVALID = "invalid"            # Garbage output (failed evaluation)
    UNSTABLE = "unstable"          # Beautiful but collapses (high prediction error later)
    STERILE = "sterile"            # Survives but produces no descendants
    CATASTROPHIC = "catastrophic"  # Destroys lineage (forbidden region)

class TransformationGenome:
    """
    A verb in the universe.
    Tracks the latent delta, its cost, and its success/failure history.
    """
    def __init__(self, id: int, delta_latent: np.ndarray, parent_id: Optional[int] = None):
        self.id = id
        self.delta_latent = delta_latent
        self.magnitude = float(np.linalg.norm(delta_latent))
        
        # Energy cost scales with magnitude. A massive jump costs more.
        self.energy_cost = self.magnitude * 1.5 
        
        self.tier = TransformationTier.EVENT
        self.confidence = 0.1
        
        # Verb Lineage
        self.parent_id = parent_id
        self.children_ids: List[int] = []
        
        # Fitness tracking (Usefulness vs Cost)
        self.success_count = 0
        self.failure_history: Dict[str, int] = {f.value: 0 for f in FailureType}
        self.total_offspring_fitness = 0.0
        self.novelty_created = 0.0
        
        # The domains (concept IDs or species IDs) where this verb works
        self.applicability_domain: Set[int] = set() 
        
        self.creation_time = time.time()
        self.last_used = time.time()

    def get_fitness(self) -> float:
        """
        Transformation fitness = (usefulness) - (cost)
        A boring but reliable verb survives. A spectacular but destructive one dies.
        """
        total_uses = self.success_count + sum(self.failure_history.values())
        if total_uses == 0:
            return 0.1
            
        success_rate = self.success_count / total_uses
        avg_offspring_quality = self.total_offspring_fitness / max(1, self.success_count)
        avg_novelty = self.novelty_created / max(1, self.success_count)
        domain_expansion = np.log1p(len(self.applicability_domain))
        
        usefulness = (avg_offspring_quality * domain_expansion * avg_novelty * success_rate)
        
        failure_rate = 1.0 - success_rate
        cost = self.energy_cost + (failure_rate * 2.0)
        
        return max(0.01, usefulness - cost)

    def mutate(self, new_id: int) -> 'TransformationGenome':
        """
        Evolve the verb itself.
        Can mutate magnitude, direction, or energy efficiency.
        """
        mutation_type = np.random.choice(["magnitude", "direction", "efficiency"])
        new_delta = self.delta_latent.copy()
        
        if mutation_type == "magnitude":
            # Scale the delta up or down
            scale = np.random.uniform(0.8, 1.2)
            new_delta *= scale
        elif mutation_type == "direction":
            # Add a small orthogonal drift
            noise = np.random.randn(*new_delta.shape).astype(np.float32)
            # Make noise orthogonal to current direction
            noise -= np.dot(noise.flatten(), new_delta.flatten()) / (np.linalg.norm(new_delta)**2 + 1e-8) * new_delta
            # NORMALIZE noise before scaling so we don't accidentally add a massive vector
            noise_norm = np.linalg.norm(noise)
            if noise_norm > 1e-8:
                noise /= noise_norm
            new_delta += noise * 0.1 * np.linalg.norm(new_delta)
            
        child_verb = TransformationGenome(new_id, new_delta, parent_id=self.id)
        
        if mutation_type == "efficiency":
            # Verb learned to do the same thing with less energy
            child_verb.energy_cost = self.energy_cost * np.random.uniform(0.7, 0.95)
            
        self.children_ids.append(new_id)
        return child_verb

    def record_success(self, domain_id: int, offspring_fitness: float, novelty: float):
        self.success_count += 1
        self.total_offspring_fitness += offspring_fitness
        self.novelty_created += novelty
        self.applicability_domain.add(domain_id)
        self.last_used = time.time()
        self._update_tier()
        
    def record_failure(self, failure_type: FailureType):
        self.failure_history[failure_type.value] += 1
        self.last_used = time.time()
        self.confidence *= 0.9 # Penalize
        self._update_tier()
        
    def _update_tier(self):
        """Promote or demote the transformation based on empirical evidence."""
        total_uses = self.success_count + sum(self.failure_history.values())
        if total_uses == 0:
            return
            
        success_rate = self.success_count / total_uses
        domain_size = len(self.applicability_domain)
        
        if self.success_count > 1000 and success_rate > 0.8 and domain_size > 50:
            self.tier = TransformationTier.UNIVERSAL_LAW
            self.confidence = 0.99
        elif self.success_count > 100 and success_rate > 0.7 and domain_size > 10:
            self.tier = TransformationTier.SPECIES_LAW
            self.confidence = 0.80
        elif self.success_count > 20 and success_rate > 0.5:
            self.tier = TransformationTier.HYPOTHESIS
            self.confidence = 0.50
        elif self.success_count > 3:
            self.tier = TransformationTier.PATTERN
            self.confidence = 0.30
        elif self.confidence < 0.05 and self.success_count > 100:
            # It was highly successful once, but confidence has decayed to nothing
            self.tier = TransformationTier.FOSSILIZED_LAW
        else:
            self.tier = TransformationTier.EVENT
            self.confidence = max(0.01, self.confidence) # Don't let it hit exactly 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "delta_latent": self.delta_latent.tolist(),
            "magnitude": self.magnitude,
            "energy_cost": self.energy_cost,
            "tier": self.tier.value,
            "confidence": self.confidence,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "success_count": self.success_count,
            "failure_history": self.failure_history,
            "total_offspring_fitness": self.total_offspring_fitness,
            "novelty_created": self.novelty_created,
            "applicability_domain": list(self.applicability_domain),
            "creation_time": self.creation_time,
            "last_used": self.last_used
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TransformationGenome':
        delta = np.array(data["delta_latent"], dtype=np.float32)
        genome = cls(data["id"], delta, data.get("parent_id"))
        genome.magnitude = data["magnitude"]
        genome.energy_cost = data["energy_cost"]
        genome.tier = TransformationTier(data["tier"])
        genome.confidence = data["confidence"]
        genome.children_ids = data.get("children_ids", [])
        genome.success_count = data["success_count"]
        genome.failure_history = data["failure_history"]
        genome.total_offspring_fitness = data.get("total_offspring_fitness", 0.0)
        genome.novelty_created = data.get("novelty_created", 0.0)
        genome.applicability_domain = set(data["applicability_domain"])
        genome.creation_time = data["creation_time"]
        genome.last_used = data["last_used"]
        return genome


class PhysicsEngine:
    """
    Observes mutations and extracts the underlying verbs (transformations).
    """
    def __init__(self, db_conn, db_lock=None):
        self.db_conn = db_conn
        self.db_lock = db_lock
        self.transformations: Dict[int, TransformationGenome] = {}
        self.similarity_threshold = 0.85 # Cosine similarity needed to match verbs
        self.next_id = 1
        self.load_from_db()

    def observe_mutation(self, parent_latent: np.ndarray, child_latent: np.ndarray, 
                         parent_id: int, success: bool, offspring_fitness: float, novelty: float, 
                         failure_type: Optional[FailureType] = None):
        """
        Extract the delta. If it matches a known verb, update the verb's history.
        If it's new and successful, register it as an Event.
        """
        delta = child_latent - parent_latent
        delta_norm = np.linalg.norm(delta)
        
        if delta_norm < 1e-4:
            return None # Too small to be a meaningful verb
            
        delta_flat = delta.flatten()
        
        # Find matching transformation (verb)
        best_match = None
        best_sim = -1
        
        for trans in self.transformations.values():
            t_flat = trans.delta_latent.flatten()
            # Cosine similarity
            sim = np.dot(delta_flat, t_flat) / (delta_norm * np.linalg.norm(t_flat) + 1e-8)
            if sim > best_sim:
                best_sim = sim
                best_match = trans
                
        if best_sim > self.similarity_threshold:
            # It's an existing verb!
            if success:
                best_match.record_success(parent_id, offspring_fitness, novelty)
            else:
                best_match.record_failure(failure_type or FailureType.INVALID)
            self.save_transformation(best_match)
            return best_match
        else:
            # It's a new event
            if success: # We only register new verbs if they succeed at least once
                new_trans = TransformationGenome(self.next_id, delta)
                self.next_id += 1
                new_trans.record_success(parent_id, offspring_fitness, novelty)
                self.transformations[new_trans.id] = new_trans
                self.save_transformation(new_trans)
                return new_trans
                
        return None

    def get_verb_candidates(self, available_energy: float = float('inf')) -> List[TransformationGenome]:
        """
        Returns a list of verbs weighted by their fitness for the DreamEngine to use.
        Filters out verbs that cost more energy than the concept has available.
        """
        if not self.transformations:
            return []
            
        # Filter by thermodynamics
        affordable_verbs = [v for v in self.transformations.values() if v.energy_cost <= available_energy]
        
        if not affordable_verbs:
            return []
            
        fitnesses = np.array([v.get_fitness() for v in affordable_verbs])
        
        # Normalize to probabilities
        if fitnesses.sum() == 0:
            probs = np.ones(len(fitnesses)) / len(fitnesses)
        else:
            probs = fitnesses / fitnesses.sum()
            
        # Return sorted by probability (highest first)
        sorted_indices = np.argsort(probs)[::-1]
        return [affordable_verbs[i] for i in sorted_indices]
        
    def decay_laws(self, current_time: float):
        """
        Phase 6: Deep Time.
        Laws that are not used slowly decay. A Universal Law can become a Fossilized Law.
        """
        for trans in self.transformations.values():
            time_since_used = current_time - trans.last_used
            if time_since_used > 500: # 500 generations of neglect
                # Decay confidence by 5%
                trans.confidence *= 0.95
                trans._update_tier()
                self.save_transformation(trans)

    def evolve_verbs(self):
        """
        Periodically mutate highly successful verbs to discover new physics.
        """
        candidates = [v for v in self.transformations.values() if v.tier in [TransformationTier.SPECIES_LAW, TransformationTier.UNIVERSAL_LAW, TransformationTier.HYPOTHESIS]]
        if not candidates:
            return
            
        # Pick a random successful verb to mutate
        parent_verb = np.random.choice(candidates)
        child_verb = parent_verb.mutate(self.next_id)
        self.next_id += 1
        
        self.transformations[child_verb.id] = child_verb
        self.save_transformation(parent_verb) # Save parent to update children_ids
        self.save_transformation(child_verb)
        print(f"[Physics] Verb {parent_verb.id} mutated into Verb {child_verb.id}")

    def extract_semantic_verb(self, latent_a: np.ndarray, latent_b: np.ndarray) -> Optional['TransformationGenome']:
        """
        Actively extracts a verb by finding the vector difference between two successful states.
        Calculates the "A-to-B" latent trajectory.
        """
        delta = latent_b - latent_a
        
        # Guard against NaN/Inf values destroying the C++ backend
        if not np.isfinite(delta).all():
            print("  [Physics] Warning: Non-finite values detected in latent extraction. Aborting.")
            return None
            
        delta_norm = float(np.linalg.norm(delta))
        
        if delta_norm < 1e-4 or not np.isfinite(delta_norm):
            return None
            
        # Normalize to a standard mutation magnitude so it doesn't instantly obliterate targets
        target_magnitude = 8.0 
        delta = (delta / delta_norm) * target_magnitude
        delta = delta.astype(np.float32)  # Ensure float32 to prevent PyTorch C++ segfaults
        
        new_trans = TransformationGenome(self.next_id, delta)
        self.next_id += 1
        
        # Fast-track it to Hypothesis tier so DreamEngine tests it immediately
        new_trans.tier = TransformationTier.HYPOTHESIS
        new_trans.confidence = 0.5
        new_trans.success_count = 1 # Fake initial success to boost priority
        
        self.transformations[new_trans.id] = new_trans
        self.save_transformation(new_trans)
        return new_trans

    def save_transformation(self, trans: TransformationGenome):
        if self.db_lock:
            with self.db_lock:
                self._save_transformation_impl(trans)
        else:
            self._save_transformation_impl(trans)
    
    def _save_transformation_impl(self, trans: TransformationGenome):
        cursor = self.db_conn.cursor()
        data = trans.to_dict()
        cursor.execute('''
            INSERT OR REPLACE INTO transformations 
            (id, delta_latent, magnitude, energy_cost, tier, confidence, 
             parent_id, children_ids, success_count, failure_history, 
             total_offspring_fitness, novelty_created, domains, creation_time, last_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data["id"],
            json.dumps(data["delta_latent"]),
            data["magnitude"],
            data["energy_cost"],
            data["tier"],
            data["confidence"],
            data["parent_id"],
            json.dumps(data["children_ids"]),
            data["success_count"],
            json.dumps(data["failure_history"]),
            data["total_offspring_fitness"],
            data["novelty_created"],
            json.dumps(data["applicability_domain"]),
            data["creation_time"],
            data["last_used"]
        ))
        self.db_conn.commit()

    def load_from_db(self):
        if self.db_lock:
            with self.db_lock:
                self._load_from_db_impl()
        else:
            self._load_from_db_impl()
    
    def _load_from_db_impl(self):
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transformations'")
        if not cursor.fetchone():
            return # Table doesn't exist yet

        cursor.execute("SELECT * FROM transformations")
        for row in cursor:
            data = {
                "id": row[0],
                "delta_latent": json.loads(row[1]),
                "magnitude": row[2],
                "energy_cost": row[3],
                "tier": row[4],
                "confidence": row[5],
                "parent_id": row[6],
                "children_ids": json.loads(row[7]) if row[7] else [],
                "success_count": row[8],
                "failure_history": json.loads(row[9]),
                "total_offspring_fitness": row[10],
                "novelty_created": row[11],
                "applicability_domain": json.loads(row[12]),
                "creation_time": row[13],
                "last_used": row[14]
            }
            trans = TransformationGenome.from_dict(data)
            self.transformations[trans.id] = trans
            if trans.id >= self.next_id:
                self.next_id = trans.id + 1
        
        print(f"Loaded {len(self.transformations)} transformations (verbs) from physics engine.")
