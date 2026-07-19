"""
Neuronal Memory System
Biologically-inspired memory with decay, competition, and synaptic strengthening.

Key Concepts:
- Concepts are neurons with activation levels
- Connections have synaptic weights that strengthen with use
- Unused concepts decay and die (apoptosis)
- Memory is limited, concepts compete for survival
- Similar concepts consolidate (memory compression)
- Hebbian learning: "neurons that fire together wire together"
"""

import numpy as np
import sqlite3
import json
import time
from typing import Dict, List, Tuple, Optional

class ConceptNeuron:
    """
    A concept neuron with activation, connections, and survival metrics.
    """
    def __init__(self, concept_id: int, embedding: np.ndarray, 
                 latent: Optional[np.ndarray], generation: int, is_real_image: bool = False):
        self.concept_id = concept_id
        self.embedding = embedding
        self.latent = latent
        self.generation = generation
        self.is_real_image = is_real_image  # Track if this is a real image anchor
        
        # Neuronal properties
        self.activation = 1.0  # Current activation level (0-1)
        self.base_strength = 1.0  # Base synaptic strength
        self.age = 0  # Time since creation
        self.last_access = time.time()  # Last retrieval time
        
        # Usage metrics
        self.access_count = 0  # How many times retrieved
        self.contribution_score = 0.0  # Contribution to new concepts
        self.descendants = 0  # Number of child concepts
        
        # Connections (synapses)
        self.connections = {}  # {concept_id: synaptic_weight}
        
    def activate(self, strength: float = 0.2):
        """Fire this neuron, increasing activation."""
        self.activation = min(1.0, self.activation + strength)
        self.last_access = time.time()
        self.access_count += 1
        
    def decay(self, rate: float = 0.01):
        """Natural decay over time."""
        self.activation *= (1.0 - rate)
        self.base_strength *= (1.0 - rate * 0.5)  # Slower decay for base strength
        
    def strengthen_connection(self, other_id: int, amount: float = 0.1):
        """Hebbian learning: strengthen synapse to another concept."""
        if other_id not in self.connections:
            self.connections[other_id] = 0.0
        self.connections[other_id] = min(1.0, self.connections[other_id] + amount)
        
    def fitness(self, current_generation: int = 0) -> float:
        """
        Calculate survival fitness.
        Combines activation, usage, contribution, and connections.
        Includes reality bias that decays over time.
        """
        # Base fitness components
        activation_score = self.activation * 0.25
        usage_score = min(1.0, self.access_count / 50.0) * 0.20
        contribution_score = min(1.0, self.contribution_score / 10.0) * 0.25
        connectivity = len(self.connections) * sum(self.connections.values())
        connectivity_score = min(1.0, connectivity / 20.0) * 0.20
        time_since_access = time.time() - self.last_access
        recency_score = max(0.0, 1.0 - (time_since_access / 86400.0)) * 0.10
        
        base_fitness = activation_score + usage_score + contribution_score + connectivity_score + recency_score
        
        # Reality bias: Real images get boosted fitness early on
        if self.is_real_image:
            if current_generation < 1000:
                reality_multiplier = 2.0  # 100% boost
            elif current_generation < 3000:
                reality_multiplier = 1.5  # 50% boost
            elif current_generation < 7000:
                reality_multiplier = 1.2  # 20% boost
            else:
                reality_multiplier = 1.0  # No bias
        else:
            reality_multiplier = 1.0
            
        return base_fitness * reality_multiplier
        
    def should_survive(self, threshold: float = 0.15) -> bool:
        """Determine if concept should survive pruning."""
        return self.fitness() > threshold


class NeuronalMemorySystem:
    """
    Memory system with neuronal dynamics.
    """
    def __init__(self, db_path: str = "universe.db", 
                 max_concepts: int = 5000,
                 decay_rate: float = 0.005):
        self.db_path = db_path
        self.max_concepts = max_concepts  # Memory capacity limit
        self.decay_rate = decay_rate
        
        self.neurons = {}  # {concept_id: ConceptNeuron}
        self.last_prune_time = time.time()
        self.prune_interval = 600.0  # Prune every 10 minutes
        
        # Competition metrics
        self.total_activations = 0
        self.pruned_count = 0
        self.consolidated_count = 0
        
    def load_from_database(self, conn: sqlite3.Connection):
        """Load concepts from database into neuronal memory."""
        cursor = conn.cursor()
        
        # Get list of real images
        cursor.execute("SELECT concept_id FROM ingested_files")
        real_image_ids = set(row[0] for row in cursor)
        
        cursor.execute("""
            SELECT id, embedding, generation, times_referenced, date_added
            FROM concepts
        """)
        
        for row in cursor:
            c_id, emb_json, gen, refs, date_added = row
            
            embedding = np.array(json.loads(emb_json)).flatten()
            latent = None # Don't load latent to save memory
            
            is_real = c_id in real_image_ids
            neuron = ConceptNeuron(c_id, embedding, latent, gen, is_real_image=is_real)
            neuron.access_count = refs
            neuron.last_access = date_added
            
            self.neurons[c_id] = neuron
            
        real_count = sum(1 for n in self.neurons.values() if n.is_real_image)
        print(f"Loaded {len(self.neurons)} concepts ({real_count} real images, {len(self.neurons)-real_count} derived)")
        
    def activate_concept(self, concept_id: int, strength: float = 0.2):
        """Activate a concept neuron."""
        if concept_id in self.neurons:
            self.neurons[concept_id].activate(strength)
            self.total_activations += 1
            
    def activate_pattern(self, embedding: np.ndarray, k: int = 5):
        """
        Activate a pattern of similar concepts (spreading activation).
        Returns activated concept IDs.
        """
        # Find nearest neighbors
        distances = []
        for c_id, neuron in self.neurons.items():
            dist = np.linalg.norm(embedding - neuron.embedding)
            distances.append((c_id, dist))
            
        distances.sort(key=lambda x: x[1])
        
        # Activate nearest neighbors with decaying strength
        activated = []
        for i, (c_id, dist) in enumerate(distances[:k]):
            strength = 0.3 * (1.0 - (i / k))  # Decay with rank
            self.activate_concept(c_id, strength)
            activated.append(c_id)
            
        return activated
        
    def strengthen_connections(self, concept_ids: List[int]):
        """
        Hebbian learning: concepts used together form stronger connections.
        """
        for i, c_id_1 in enumerate(concept_ids):
            if c_id_1 not in self.neurons:
                continue
            for c_id_2 in concept_ids[i+1:]:
                if c_id_2 not in self.neurons:
                    continue
                    
                # Bidirectional strengthening
                self.neurons[c_id_1].strengthen_connection(c_id_2, 0.05)
                self.neurons[c_id_2].strengthen_connection(c_id_1, 0.05)
                
    def record_contribution(self, parent_ids: List[int], child_id: int, quality: float):
        """Record that parents contributed to creating a child."""
        for parent_id in parent_ids:
            if parent_id in self.neurons:
                self.neurons[parent_id].contribution_score += quality
                self.neurons[parent_id].descendants += 1
                
        # Strengthen connections between parents
        self.strengthen_connections(parent_ids)
        
    def apply_decay(self):
        """Apply natural decay to all neurons."""
        for neuron in self.neurons.values():
            neuron.decay(self.decay_rate)
            
    def consolidate_similar(self, similarity_threshold: float = 0.98, max_consolidations: int = 100):
        """
        Memory consolidation: merge very similar concepts.
        Like memory replay during sleep.
        Only consolidates extremely similar concepts to prevent over-merging.
        """
        consolidated = []
        
        neurons_list = list(self.neurons.values())
        
        # Only check subset to avoid O(n²) explosion
        sample_size = min(500, len(neurons_list))
        if len(neurons_list) > sample_size:
            # Sample random neurons for efficiency
            import random
            neurons_list = random.sample(neurons_list, sample_size)
        
        for i, neuron1 in enumerate(neurons_list):
            if len(consolidated) >= max_consolidations:
                break
                
            for neuron2 in neurons_list[i+1:]:
                if neuron2.concept_id in consolidated:
                    continue
                    
                # Check similarity (both must be normalized)
                similarity = np.dot(neuron1.embedding, neuron2.embedding)
                
                if similarity > similarity_threshold:
                    # Merge weaker into stronger
                    if neuron1.fitness() >= neuron2.fitness():
                        keeper, merged = neuron1, neuron2
                    else:
                        keeper, merged = neuron2, neuron1
                        
                    # Transfer properties
                    keeper.activation = max(keeper.activation, merged.activation)
                    keeper.access_count += merged.access_count
                    keeper.contribution_score += merged.contribution_score
                    keeper.descendants += merged.descendants
                    
                    # Merge connections
                    for c_id, weight in merged.connections.items():
                        if c_id in keeper.connections:
                            keeper.connections[c_id] = max(keeper.connections[c_id], weight)
                        else:
                            keeper.connections[c_id] = weight
                            
                    consolidated.append(merged.concept_id)
                    
        # Remove merged concepts
        for c_id in consolidated:
            if c_id in self.neurons:
                del self.neurons[c_id]
                
        self.consolidated_count += len(consolidated)
        return consolidated
        
    def prune_weak_concepts(self, fitness_threshold: float = 0.15):
        """
        Apoptosis: remove weak, unused concepts.
        Like synaptic pruning during sleep.
        """
        to_prune = []
        
        for c_id, neuron in self.neurons.items():
            if not neuron.should_survive(fitness_threshold):
                to_prune.append(c_id)
                
        # Remove weakest concepts
        for c_id in to_prune:
            del self.neurons[c_id]
            
        self.pruned_count += len(to_prune)
        return to_prune
        
    def enforce_capacity_limit(self):
        """
        Competition: if memory is full, remove weakest concepts.
        """
        if len(self.neurons) <= self.max_concepts:
            return []
            
        # Sort by fitness
        neurons_by_fitness = sorted(
            self.neurons.items(),
            key=lambda x: x[1].fitness(),
            reverse=True
        )
        
        # Keep only top max_concepts
        to_remove = [c_id for c_id, _ in neurons_by_fitness[self.max_concepts:]]
        
        for c_id in to_remove:
            del self.neurons[c_id]
            
        self.pruned_count += len(to_remove)
        return to_remove
        
    def maintenance_cycle(self):
        """
        Periodic maintenance: decay, consolidation, pruning.
        Like memory consolidation during sleep.
        """
        print("\n" + "="*60)
        print("NEURONAL MEMORY MAINTENANCE")
        print("="*60)
        
        initial_count = len(self.neurons)
        
        # 1. Apply decay
        self.apply_decay()
        print(f"Applied decay (rate={self.decay_rate})")
        
        # 2. Consolidate similar concepts (only extremely similar, cap at 100)
        consolidated = self.consolidate_similar(similarity_threshold=0.98, max_consolidations=100)
        print(f"Consolidated {len(consolidated)} similar concepts")
        
        # 3. Prune weak concepts
        pruned = self.prune_weak_concepts(fitness_threshold=0.15)
        print(f"Pruned {len(pruned)} weak concepts")
        
        # 4. Enforce capacity limit (competition)
        removed = self.enforce_capacity_limit()
        print(f"Removed {len(removed)} to enforce capacity limit")
        
        final_count = len(self.neurons)
        print(f"\nMemory: {initial_count} -> {final_count} concepts")
        print(f"Total pruned (lifetime): {self.pruned_count}")
        print(f"Total consolidated: {self.consolidated_count}")
        print("="*60)
        
        self.last_prune_time = time.time()
        
        return {
            "initial": initial_count,
            "final": final_count,
            "consolidated": len(consolidated),
            "pruned": len(pruned),
            "capacity_removed": len(removed)
        }
        
    def should_run_maintenance(self) -> bool:
        """Check if it's time for maintenance."""
        time_based = (time.time() - self.last_prune_time) > self.prune_interval
        capacity_based = len(self.neurons) > (self.max_concepts * 0.8)  # 80% full
        return time_based or capacity_based
        
    def get_top_concepts(self, k: int = 10, current_generation: int = 0) -> List[Tuple[int, float]]:
        """Get top-k concepts by fitness (with reality bias if early)."""
        neurons_by_fitness = sorted(
            self.neurons.items(),
            key=lambda x: x[1].fitness(current_generation),
            reverse=True
        )
        return [(c_id, n.fitness(current_generation)) for c_id, n in neurons_by_fitness[:k]]
        
    def get_status(self) -> Dict:
        """Get memory system status."""
        if not self.neurons:
            return {
                "total_concepts": 0,
                "avg_fitness": 0.0,
                "capacity_used": 0.0
            }
            
        fitnesses = [n.fitness() for n in self.neurons.values()]
        activations = [n.activation for n in self.neurons.values()]
        
        return {
            "total_concepts": len(self.neurons),
            "capacity_used": len(self.neurons) / self.max_concepts,
            "avg_fitness": np.mean(fitnesses),
            "avg_activation": np.mean(activations),
            "total_connections": sum(len(n.connections) for n in self.neurons.values()),
            "total_activations": self.total_activations,
            "pruned_lifetime": self.pruned_count,
            "consolidated_lifetime": self.consolidated_count
        }


# Integration example
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    
    print("Testing Neuronal Memory System")
    print("="*60)
    
    # Create system
    memory = NeuronalMemorySystem(max_concepts=100, decay_rate=0.01)
    
    # Load from database
    import sqlite3
    conn = sqlite3.connect("universe.db")
    memory.load_from_database(conn)
    conn.close()
    
    # Show initial status
    status = memory.get_status()
    print(f"\nInitial Status:")
    print(f"  Concepts: {status['total_concepts']}")
    print(f"  Capacity: {status['capacity_used']*100:.1f}%")
    print(f"  Avg Fitness: {status['avg_fitness']:.3f}")
    
    # Simulate some activations
    print("\nSimulating usage patterns...")
    for i in range(50):
        # Random activation pattern
        random_embedding = np.random.randn(512).astype(np.float32)
        random_embedding = random_embedding / np.linalg.norm(random_embedding)
        memory.activate_pattern(random_embedding, k=3)
        
    # Run maintenance
    result = memory.maintenance_cycle()
    
    # Show final status
    status = memory.get_status()
    print(f"\nFinal Status:")
    print(f"  Concepts: {status['total_concepts']}")
    print(f"  Capacity: {status['capacity_used']*100:.1f}%")
    print(f"  Avg Fitness: {status['avg_fitness']:.3f}")
    print(f"  Total Activations: {status['total_activations']}")
    
    # Show top concepts
    print("\nTop 10 Concepts by Fitness:")
    for c_id, fitness in memory.get_top_concepts(10):
        print(f"  Concept {c_id}: fitness={fitness:.3f}")
        
    print("\n[OK] Neuronal memory system operational")
