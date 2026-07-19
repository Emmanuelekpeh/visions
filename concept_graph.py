"""
Concept Graph Memory System
A heterogeneous graph of concepts with typed relationships.

The memory is not a flat pool - it's a living web of:
- Lineages (evolved_from)
- Similarities (similar_to)
- Combinations (combines_with)
- Contradictions (incompatible_with)
- Specializations (specializes_into)

Three layers:
- Reality Layer: Original images (confidence=1.0, roots)
- Concept Layer: Proven derived concepts (0.1-0.9)
- Dream Layer: Experimental concepts (unknown)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
import time

class NodeTier(Enum):
    """Three-layer memory structure."""
    REALITY = "reality"      # Original images (roots)
    CONCEPT = "concept"      # Proven derived concepts
    DREAM = "dream"          # Experimental, unproven

class EdgeType(Enum):
    """Typed relationships between concepts."""
    EVOLVED_FROM = "evolved_from"          # Parent → Child (lineage)
    
    # SIMILARITY (split by type - not all similarity is the same!)
    VISUAL_SIMILAR = "visual_similar"      # CLIP/visual similarity
    LATENT_SIMILAR = "latent_similar"      # VAE latent space similarity
    FUNCTIONAL_SIMILAR = "functional_similar"  # Behavioral similarity
    
    # COMBINATION & CONTRADICTION
    COMBINES_WITH = "combines_with"        # Successful recombination
    CONTRADICTS = "contradicts"            # Incompatible concepts
    
    # HAZARD MAP (Failures)
    FAILED_INVALID = "failed_invalid"      # Immediate garbage output
    FAILED_UNSTABLE = "failed_unstable"    # Collapsed after a few generations
    FAILED_STERILE = "failed_sterile"      # Survived but produced no descendants
    FAILED_CATASTROPHIC = "failed_catastrophic" # Destroyed lineage (forbidden region)
    
    # HIERARCHY
    SPECIALIZES_INTO = "specializes_into"  # Generalization → Specialization
    
    # TEMPORAL (world processes, not just objects!)
    TRANSFORMED_INTO = "transformed_into"  # A → B transformation (tree → crystal_tree → glass_tree)
    CAUSED_BY = "caused_by"                # Causal relationship

class ConceptEcology:
    """
    Phase 6: Scarcity.
    Tracks the resource footprint and ecological value of a concept.
    """
    def __init__(self, complexity: float):
        # High complexity (e.g., massive alien cathedral) costs more to maintain
        self.energy_cost = max(0.1, min(12.0, complexity * 2.0))
        self.memory_cost = 1.0 # Base cost to exist in the graph
        
        # How much attention it consumes from the observer/system
        self.attention_consumption = 0.0 
        
        # How much it contributes back to the ecosystem
        self.ecological_value = 0.0
        
        # Redundancy (how many highly similar neighbors it has)
        self.redundancy = 0.0

    def to_dict(self) -> dict:
        return {
            "energy_cost": self.energy_cost,
            "memory_cost": self.memory_cost,
            "attention_consumption": self.attention_consumption,
            "ecological_value": self.ecological_value,
            "redundancy": self.redundancy
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ConceptEcology':
        c = cls(0.5) # Dummy complexity, overridden below
        c.energy_cost = data.get("energy_cost", 1.0)
        c.memory_cost = data.get("memory_cost", 1.0)
        c.attention_consumption = data.get("attention_consumption", 0.0)
        c.ecological_value = data.get("ecological_value", 0.0)
        c.redundancy = data.get("redundancy", 0.0)
        return c

@dataclass
class ConceptNode:
    """
    A node in the concept graph.
    Stores latent, embedding, image, confidence, and lineage.
    """
    id: int
    embedding: np.ndarray
    tier: NodeTier
    confidence: float
    age: int  # Generations since creation
    fitness: float
    latent: Optional[np.ndarray] = None
    
    # Graph relationships
    parents: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    
    # Similarity (typed - visual, latent, functional)
    visual_similar: List[Tuple[int, float]] = field(default_factory=list)
    latent_similar: List[Tuple[int, float]] = field(default_factory=list)
    functional_similar: List[Tuple[int, float]] = field(default_factory=list)
    
    # Combination & contradiction
    combines_with: List[Tuple[int, int, float]] = field(default_factory=list)  # (node_a, node_b, success_score)
    contradicts: List[int] = field(default_factory=list)
    
    # Hazard Map (Failures)
    failed_invalid: List[int] = field(default_factory=list)      # Nodes that failed immediately from this
    failed_unstable: List[int] = field(default_factory=list)     # Nodes that collapsed
    failed_sterile: List[int] = field(default_factory=list)      # Nodes that were dead ends
    failed_catastrophic: List[int] = field(default_factory=list) # Nodes that destroyed lineages
    
    # Hierarchy
    specializes_into: List[int] = field(default_factory=list)
    
    # Temporal (processes!)
    transformed_into: List[int] = field(default_factory=list)  # This node → next state
    caused_by: List[int] = field(default_factory=list)  # What caused this node
    
    # Validation
    creation_time: float = field(default_factory=time.time)
    last_validation: float = field(default_factory=time.time)
    validation_count: int = 0
    prediction_errors: List[float] = field(default_factory=list)
    reality_distance: float = 999.0  # Distance to nearest reality node
    
    # Phase 6: Scarcity & Ecology
    ecology: ConceptEcology = field(default_factory=lambda: ConceptEcology(0.5))
    
    # Usage tracking
    selection_count: int = 0
    offspring_count: int = 0
    successful_offspring: int = 0
    
    # Exploration tracking (prevent GNN from becoming a cage!)
    last_visited: float = field(default_factory=time.time)
    visit_frequency: float = 0.0  # Visits per generation
    region_explored: bool = False  # Has neighborhood been explored?
    
    def mean_prediction_error(self) -> float:
        """Average prediction error (learning signal)."""
        if not self.prediction_errors:
            return 0.0
        return float(np.mean(self.prediction_errors[-10:]))
        
    def offspring_success_rate(self) -> float:
        """Fraction of offspring that were successful."""
        if self.offspring_count == 0:
            return 0.0
        return self.successful_offspring / self.offspring_count
        
    def exploration_bonus(self, current_time: float) -> float:
        """
        Compute exploration bonus to prevent getting stuck in comfortable regions.
        
        Bonus is high for:
        - Old concepts that haven't been visited recently
        - Low visit frequency (unexplored)
        - Neighborhoods that haven't been fully explored
        
        This is the "dark matter" exploration budget.
        """
        # Time since last visit (normalized)
        time_since_visit = max(0.0, current_time - self.last_visited)  # Ensure non-negative
        time_bonus = min(1.0, time_since_visit / 1000.0)  # Cap at 1.0
        
        # Inverse visit frequency (less visited = higher bonus)
        frequency_bonus = max(0.0, 1.0 - self.visit_frequency)
        
        # Region exploration bonus
        region_bonus = 0.0 if self.region_explored else 0.3
        
        # Combine (all components are non-negative)
        total_bonus = 0.5 * time_bonus + 0.3 * frequency_bonus + 0.2 * region_bonus
        
        return float(max(0.0, total_bonus))  # Final safety check


class ConceptGraph:
    """
    Heterogeneous concept graph with typed edges and three-tier structure.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.nodes: Dict[int, ConceptNode] = {}
        self.similarity_threshold = similarity_threshold
        
        # Index by tier for fast lookup
        self.reality_nodes: Set[int] = set()
        self.concept_nodes: Set[int] = set()
        self.dream_nodes: Set[int] = set()
        
        # Statistics
        self.total_promotions = 0
        self.total_demotions = 0
        self.total_quarantines = 0
        
    def add_node(self, node: ConceptNode):
        """Add a node to the graph and update indices."""
        self.nodes[node.id] = node
        
        # Update tier index
        if node.tier == NodeTier.REALITY:
            self.reality_nodes.add(node.id)
        elif node.tier == NodeTier.CONCEPT:
            self.concept_nodes.add(node.id)
        elif node.tier == NodeTier.DREAM:
            self.dream_nodes.add(node.id)
            
    def add_edge(self, source_id: int, target_id: int, edge_type: EdgeType, weight: float = 1.0):
        """Add a typed edge between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return
            
        source = self.nodes[source_id]
        target = self.nodes[target_id]
        
        if edge_type == EdgeType.EVOLVED_FROM:
            target.parents.append(source_id)
            source.children.append(target_id)
            source.offspring_count += 1
            
        elif edge_type == EdgeType.VISUAL_SIMILAR or edge_type == EdgeType.LATENT_SIMILAR or edge_type == EdgeType.FUNCTIONAL_SIMILAR:
            if edge_type == EdgeType.VISUAL_SIMILAR:
                source.visual_similar.append((target_id, weight))
                target.visual_similar.append((source_id, weight))
            elif edge_type == EdgeType.LATENT_SIMILAR:
                source.latent_similar.append((target_id, weight))
                target.latent_similar.append((source_id, weight))
            elif edge_type == EdgeType.FUNCTIONAL_SIMILAR:
                source.functional_similar.append((target_id, weight))
                target.functional_similar.append((source_id, weight))
            
        elif edge_type == EdgeType.COMBINES_WITH:
            # For combines_with, weight is the success score of the combination
            # Store as (parent_a, parent_b, success)
            # This is stored on the child node
            pass  # Handled in add_combination_edge
            
        elif edge_type == EdgeType.CONTRADICTS:
            source.contradicts.append(target_id)
            target.contradicts.append(source_id)
            
        elif edge_type == EdgeType.FAILED_INVALID:
            source.failed_invalid.append(target_id)
            
        elif edge_type == EdgeType.FAILED_UNSTABLE:
            source.failed_unstable.append(target_id)
            
        elif edge_type == EdgeType.FAILED_STERILE:
            source.failed_sterile.append(target_id)
            
        elif edge_type == EdgeType.FAILED_CATASTROPHIC:
            source.failed_catastrophic.append(target_id)
            
        elif edge_type == EdgeType.SPECIALIZES_INTO:
            source.specializes_into.append(target_id)
            
    def add_combination_edge(self, parent_a: int, parent_b: int, child_id: int, success_score: float):
        """
        Record a successful combination.
        This teaches the GNN: "parent_a + parent_b → good offspring"
        """
        if child_id not in self.nodes:
            return
            
        child = self.nodes[child_id]
        child.combines_with.append((parent_a, parent_b, success_score))
        
        # Mark parents as having successful offspring if score is high
        if success_score > 0.6:
            if parent_a in self.nodes:
                self.nodes[parent_a].successful_offspring += 1
            if parent_b in self.nodes:
                self.nodes[parent_b].successful_offspring += 1
                
    def compute_similarity_edges(self, node_id: int, k: int = 10):
        """
        Compute similarity edges for a node to its k-nearest neighbors.
        """
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        # Compute distances to all other nodes
        distances = []
        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue
            dist = np.linalg.norm(node.embedding - other_node.embedding)
            similarity = max(0.0, 1.0 - (dist / 2.0))  # Normalize to [0, 1]
            
            if similarity > self.similarity_threshold:
                distances.append((other_id, similarity))
                
        # Sort by similarity
        distances.sort(key=lambda x: x[1], reverse=True)
        
        # Add top-k edges
        for other_id, sim in distances[:k]:
            self.add_edge(node_id, other_id, EdgeType.VISUAL_SIMILAR, weight=sim)
            
    def compute_reality_distance(self, node_id: int) -> float:
        """
        Compute distance to nearest reality node.
        This is the "return-to-reality test".
        """
        if node_id not in self.nodes:
            return 999.0
            
        node = self.nodes[node_id]
        
        # If this IS a reality node, distance is 0
        if node.tier == NodeTier.REALITY:
            return 0.0
            
        # Find nearest reality node
        min_dist = float('inf')
        for reality_id in self.reality_nodes:
            reality_node = self.nodes[reality_id]
            dist = np.linalg.norm(node.embedding - reality_node.embedding)
            min_dist = min(min_dist, dist)
            
        return float(min_dist)
        
    def promote_node(self, node_id: int):
        """Promote node up one tier (Dream → Concept → stays Concept)."""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        if node.tier == NodeTier.DREAM:
            # Remove from dream
            self.dream_nodes.discard(node_id)
            # Add to concept
            node.tier = NodeTier.CONCEPT
            self.concept_nodes.add(node_id)
            self.total_promotions += 1
            
    def demote_node(self, node_id: int):
        """Demote node down one tier (Concept → Dream)."""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        if node.tier == NodeTier.CONCEPT:
            # Remove from concept
            self.concept_nodes.discard(node_id)
            # Add to dream
            node.tier = NodeTier.DREAM
            self.dream_nodes.add(node_id)
            self.total_demotions += 1
            
    def quarantine_node(self, node_id: int):
        """Remove a dangerous node from the graph."""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        # Remove from tier index
        self.reality_nodes.discard(node_id)
        self.concept_nodes.discard(node_id)
        self.dream_nodes.discard(node_id)
        
        # Remove from graph
        del self.nodes[node_id]
        self.total_quarantines += 1
        
    def get_promotion_candidates(self, min_validations: int = 3, min_confidence: float = 0.50) -> List[int]:
        """
        Find dream nodes eligible for promotion to concept tier.
        """
        candidates = []
        
        for node_id in self.dream_nodes:
            node = self.nodes[node_id]
            
            if (node.validation_count >= min_validations and 
                node.confidence > min_confidence and
                node.reality_distance < 1.5):  # Increased from 1.2
                candidates.append(node_id)
                
        return candidates
        
    def get_quarantine_candidates(self, max_reality_distance: float = 1.8, min_confidence: float = 0.2) -> List[int]:
        """
        Find nodes that have drifted too far or are unreliable.
        """
        candidates = []
        
        for node_id in self.dream_nodes:
            node = self.nodes[node_id]
            
            if node.reality_distance > max_reality_distance or node.confidence < min_confidence:
                candidates.append(node_id)
                
        return candidates
        
    def get_successful_combinations(self, min_score: float = 0.7) -> List[Tuple[int, int, float]]:
        """
        Get list of parent pairs that produced successful offspring.
        This is training data for the GNN.
        """
        combinations = []
        
        for node in self.nodes.values():
            for parent_a, parent_b, score in node.combines_with:
                if score >= min_score:
                    combinations.append((parent_a, parent_b, score))
                    
        return combinations
        
    def get_status(self) -> Dict:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "reality_nodes": len(self.reality_nodes),
            "concept_nodes": len(self.concept_nodes),
            "dream_nodes": len(self.dream_nodes),
            "total_promotions": self.total_promotions,
            "total_demotions": self.total_demotions,
            "total_quarantines": self.total_quarantines,
            "avg_confidence": np.mean([n.confidence for n in self.nodes.values()]) if self.nodes else 0.0,
            "avg_reality_distance": np.mean([n.reality_distance for n in self.nodes.values()]) if self.nodes else 0.0
        }
        
    def save_to_database(self, db_conn):
        """
        Save concept graph metadata to database for persistence.
        """
        import sqlite3
        import json
        
        cursor = db_conn.cursor()
        
        print("Saving concept graph metadata...")
        saved_count = 0
        
        # Create snapshot to avoid RuntimeError: dictionary changed size during iteration
        nodes_snapshot = list(self.nodes.items())
        
        for node_id, node in nodes_snapshot:
            # Prepare JSON fields
            prediction_errors_json = json.dumps(node.prediction_errors[-50:])  # Keep last 50
            combines_with_json = json.dumps(node.combines_with)
            contradicts_json = json.dumps(node.contradicts)
            specializes_into_json = json.dumps(node.specializes_into)
            transformed_into_json = json.dumps(node.transformed_into)
            caused_by_json = json.dumps(node.caused_by)
            
            # Hazard Map
            failed_invalid_json = json.dumps(node.failed_invalid)
            failed_unstable_json = json.dumps(node.failed_unstable)
            failed_sterile_json = json.dumps(node.failed_sterile)
            failed_catastrophic_json = json.dumps(node.failed_catastrophic)
            ecology_json = json.dumps(node.ecology.to_dict())
            
            # Insert or replace
            cursor.execute('''
                INSERT OR REPLACE INTO graph_metadata 
                (concept_id, tier, confidence, fitness, validation_count, prediction_errors,
                 selection_count, offspring_count, successful_offspring, last_visited,
                 visit_frequency, region_explored, reality_distance, combines_with,
                 contradicts, specializes_into, transformed_into, caused_by,
                 failed_invalid, failed_unstable, failed_sterile, failed_catastrophic, ecology)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node_id,
                node.tier.value,
                node.confidence,
                node.fitness,
                node.validation_count,
                prediction_errors_json,
                node.selection_count,
                node.offspring_count,
                node.successful_offspring,
                node.last_visited,
                node.visit_frequency,
                1 if node.region_explored else 0,
                node.reality_distance,
                combines_with_json,
                contradicts_json,
                specializes_into_json,
                transformed_into_json,
                caused_by_json,
                failed_invalid_json,
                failed_unstable_json,
                failed_sterile_json,
                failed_catastrophic_json,
                ecology_json
            ))
            saved_count += 1
            
        db_conn.commit()
        print(f"Saved metadata for {saved_count} nodes")
        
        # Update graph statistics
        cursor.execute('''
            INSERT OR REPLACE INTO history (id, concept_id, event_type, timestamp)
            VALUES (1, 0, ?, ?)
        ''', (f"graph_save: promotions={self.total_promotions}, quarantines={self.total_quarantines}", 
              time.time()))
        db_conn.commit()
    
    def load_from_database(self, db_conn):
        """
        Load existing concepts from database into graph.
        """
        import sqlite3
        import json
        
        cursor = db_conn.cursor()
        
        # Get real image IDs (reality layer)
        cursor.execute("SELECT concept_id FROM ingested_files")
        real_image_ids = {row[0] for row in cursor}
        
        # Load all concepts
        print("Executing concepts query...")
        cursor.execute("SELECT id, embedding, parent_ids, generation FROM concepts")
        
        reality_count = 0
        concept_count = 0
        dream_count = 0
        
        print("Fetching concepts...")
        
        for i, row in enumerate(cursor):
            if i % 500 == 0:
                print(f"Processed {i} concepts...")
            concept_id = row[0]
            embedding_json = row[1]
            parent_ids_json = row[2]
            generation = row[3]
            
            try:
                embedding_np = np.array(json.loads(embedding_json), dtype=np.float32)
                parent_ids = json.loads(parent_ids_json) if parent_ids_json else []
                
                # Determine tier
                if concept_id in real_image_ids:
                    tier = NodeTier.REALITY
                    confidence = 1.0
                    reality_count += 1
                elif generation < 100:  # Young concepts start in concept tier
                    tier = NodeTier.CONCEPT
                    confidence = 0.5
                    concept_count += 1
                else:  # Older unproven concepts go to dream
                    tier = NodeTier.DREAM
                    confidence = 0.3
                    dream_count += 1
                    
                # Create node
                node = ConceptNode(
                    id=concept_id,
                    embedding=embedding_np,
                    tier=tier,
                    confidence=confidence,
                    age=generation,
                    fitness=0.5,  # Will be computed
                    parents=[]    # Let add_edge populate this!
                )
                
                self.add_node(node)
                
                # Add parent edges
                for parent_id in parent_ids:
                    if parent_id in self.nodes:
                        self.add_edge(parent_id, concept_id, EdgeType.EVOLVED_FROM)
            except Exception as e:
                print(f"[WARN] Failed to load concept {concept_id}: {e}")
                continue
                
        print(f"Loaded {reality_count} reality, {concept_count} concept, {dream_count} dream nodes")
        print("Loading graph metadata...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_metadata'")
        if cursor.fetchone():
            print("Graph metadata table found. Fetching...")
            cursor.execute("SELECT * FROM graph_metadata")
            
            metadata_loaded = 0
            for i, row in enumerate(cursor):
                if i % 500 == 0:
                    print(f"Processed {i} metadata rows...")
                concept_id = row[0]
                if concept_id in self.nodes:
                    node = self.nodes[concept_id]
                    
                    # Restore metadata
                    node.tier = NodeTier(row[1]) if row[1] else node.tier
                    node.confidence = row[2] if row[2] is not None else node.confidence
                    node.fitness = row[3] if row[3] is not None else node.fitness
                    node.validation_count = row[4] if row[4] is not None else 0
                    
                    # Restore JSON fields
                    try:
                        node.prediction_errors = json.loads(row[5]) if row[5] else []
                        node.combines_with = json.loads(row[13]) if row[13] else []
                        node.contradicts = json.loads(row[14]) if row[14] else []
                        node.specializes_into = json.loads(row[15]) if row[15] else []
                        node.transformed_into = json.loads(row[16]) if row[16] else []
                        node.caused_by = json.loads(row[17]) if row[17] else []
                        
                        # Handle new hazard map columns if they exist in the DB schema
                        if len(row) > 18:
                            node.failed_invalid = json.loads(row[18]) if row[18] else []
                            node.failed_unstable = json.loads(row[19]) if row[19] else []
                            node.failed_sterile = json.loads(row[20]) if row[20] else []
                            node.failed_catastrophic = json.loads(row[21]) if row[21] else []
                        if len(row) > 22:
                            node.ecology = ConceptEcology.from_dict(json.loads(row[22])) if row[22] else ConceptEcology(0.5)
                    except (json.JSONDecodeError, TypeError, IndexError):
                        pass
                    
                    node.selection_count = row[6] if row[6] is not None else 0
                    node.offspring_count = row[7] if row[7] is not None else 0
                    node.successful_offspring = row[8] if row[8] is not None else 0
                    node.last_visited = row[9] if row[9] is not None else time.time()
                    node.visit_frequency = row[10] if row[10] is not None else 0.0
                    node.region_explored = bool(row[11]) if row[11] is not None else False
                    node.reality_distance = row[12] if row[12] is not None else 999.0
                    
                    # Update tier indices
                    if node.tier == NodeTier.REALITY:
                        self.reality_nodes.add(concept_id)
                    elif node.tier == NodeTier.CONCEPT:
                        self.concept_nodes.add(concept_id)
                        self.dream_nodes.discard(concept_id)  # Remove from dream if promoted
                    elif node.tier == NodeTier.DREAM:
                        self.dream_nodes.add(concept_id)
                    
                    metadata_loaded += 1
            
            print(f"Restored metadata for {metadata_loaded} nodes")
        else:
            print("No graph metadata found (first run or old database)")
        
        # Remove duplicate add_node call that was in original code
        # (node was being added twice - once at line 433 and once at line 436)
                
        print(f"Loaded {reality_count} reality, {concept_count} concept, {dream_count} dream nodes")
        
        # Compute reality distances for all non-reality nodes (VECTORIZED for performance)
        print("Computing reality distances...")
        if self.reality_nodes:
            # Collect all reality embeddings
            reality_embeddings = np.array([self.nodes[rid].embedding for rid in self.reality_nodes])
            
            # Compute distances for all non-reality nodes at once
            non_reality_ids = list(self.concept_nodes) + list(self.dream_nodes)
            for node_id in non_reality_ids:
                node = self.nodes[node_id]
                # Vectorized distance computation
                distances = np.linalg.norm(reality_embeddings - node.embedding, axis=1)
                node.reality_distance = float(np.min(distances))
        
        print(f"Reality distance computation complete.")
            
        return reality_count, concept_count, dream_count


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("CONCEPT GRAPH MEMORY SYSTEM")
    print("=" * 60)
    
    graph = ConceptGraph()
    
    # Simulate adding nodes
    print("\nCreating test graph...")
    
    # Reality node (original image)
    reality_node = ConceptNode(
        id=1,
        embedding=np.random.randn(512),
        tier=NodeTier.REALITY,
        confidence=1.0,
        age=0,
        fitness=1.0,
        latent=np.random.randn(4, 64, 64)
    )
    graph.add_node(reality_node)
    
    # Dream nodes (evolved from reality)
    for i in range(2, 10):
        dream_node = ConceptNode(
            id=i,
            embedding=np.random.randn(512),
            tier=NodeTier.DREAM,
            confidence=0.4,
            age=i,
            fitness=0.5,
            parents=[1],
            latent=np.random.randn(4, 64, 64)
        )
        graph.add_node(dream_node)
        graph.add_edge(1, i, EdgeType.EVOLVED_FROM)
        
    print(f"  Created {len(graph.nodes)} nodes")
    
    # Test promotion
    print("\nTesting promotion...")
    graph.nodes[2].validation_count = 6
    graph.nodes[2].confidence = 0.75
    graph.nodes[2].reality_distance = 0.5
    
    candidates = graph.get_promotion_candidates()
    print(f"  Promotion candidates: {candidates}")
    
    if candidates:
        graph.promote_node(candidates[0])
        print(f"  Promoted node {candidates[0]} to CONCEPT tier")
        
    # Status
    print("\nGraph status:")
    status = graph.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
        
    print("\n[Concept Graph System Ready]")
