"""
Ecology Engine
Defines the biological and cultural layer of the universe.

A species is not just a graph cluster. It is a population with:
1. Graph cohesion (Louvain)
2. Shared ancestry (lineage overlap)
3. Shared transformation vocabulary (dominant verbs)
4. Ecological interaction (relationships with other species)
"""

import numpy as np
import json
import time
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx

class SpeciesTier(Enum):
    PROTO_SPECIES = "proto_species"    # Emerging cluster, unproven
    SPECIES = "species"                # Established population
    ANCIENT_SPECIES = "ancient"        # Frozen in time (speciated)
    EXTINCT = "extinct"                # No living members
    FORGOTTEN = "forgotten"            # Extinct and failed to become a legend

class LegendNode:
    """
    A cultural artifact created by the Historian.
    Not a perfect compression, but a distorted, mythologized memory of a species.
    """
    def __init__(self, id: int, true_origin_species_id: int):
        self.id = id
        self.true_origin_species_id = true_origin_species_id
        
        # The Historian's interpretation
        self.remembered_name: str = ""
        self.accuracy: float = 1.0 # 0.0 to 1.0
        
        # Compressed/distorted traits
        self.dominant_verbs: List[int] = []
        self.famous_events: List[str] = []
        self.descendant_species: List[int] = []
        
        # How much this myth influences current cultures
        self.mythic_weight: float = 0.0
        
        # The archetype (a distorted latent vector representing the "ideal" of this species)
        self.archetype_latent: Optional[List[float]] = None
        
        self.creation_time = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "true_origin_species_id": self.true_origin_species_id,
            "remembered_name": self.remembered_name,
            "accuracy": self.accuracy,
            "dominant_verbs": self.dominant_verbs,
            "famous_events": self.famous_events,
            "descendant_species": self.descendant_species,
            "mythic_weight": self.mythic_weight,
            "archetype_latent": self.archetype_latent,
            "creation_time": self.creation_time
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LegendNode':
        l = cls(data["id"], data["true_origin_species_id"])
        l.remembered_name = data.get("remembered_name", "")
        l.accuracy = data.get("accuracy", 1.0)
        l.dominant_verbs = data.get("dominant_verbs", [])
        l.famous_events = data.get("famous_events", [])
        l.descendant_species = data.get("descendant_species", [])
        l.mythic_weight = data.get("mythic_weight", 0.0)
        l.archetype_latent = data.get("archetype_latent")
        l.creation_time = data.get("creation_time", time.time())
        return l

class Culture:
    """
    The inherited mythology and biases of a species.
    Culture warps global physics for local populations.
    """
    def __init__(self):
        # Biases for the 6 RL heads (e.g., {"Beauty": 1.2, "Coherence": 0.8})
        self.aesthetic_biases: Dict[str, float] = {}
        
        # Verbs this species loves to use
        self.preferred_transformations: Set[int] = set()
        
        # Verbs this species avoids (perhaps due to past catastrophes)
        self.feared_transformations: Set[int] = set()
        
        # The mythic origin (could be hallucinated by the Historian later)
        self.origin_legend: Optional[str] = None
        
        # Toxic concepts or combinations that are culturally forbidden
        self.taboos: Set[int] = set()

    def to_dict(self) -> dict:
        return {
            "aesthetic_biases": self.aesthetic_biases,
            "preferred_transformations": list(self.preferred_transformations),
            "feared_transformations": list(self.feared_transformations),
            "origin_legend": self.origin_legend,
            "taboos": list(self.taboos)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Culture':
        c = cls()
        c.aesthetic_biases = data.get("aesthetic_biases", {})
        c.preferred_transformations = set(data.get("preferred_transformations", []))
        c.feared_transformations = set(data.get("feared_transformations", []))
        c.origin_legend = data.get("origin_legend")
        c.taboos = set(data.get("taboos", []))
        return c

class Species:
    """
    A first-class entity representing a distinct population.
    """
    def __init__(self, id: int, origin_node: int):
        self.id = id
        self.tier = SpeciesTier.PROTO_SPECIES
        
        self.origin_node = origin_node
        self.members: Set[int] = set()
        
        # The "physics" of this species: frequencies of verb usage
        self.genome_signature: Dict[int, float] = {}
        
        self.culture = Culture()
        
        # Ecological relationships: species_id -> interaction weight
        # Positive = symbiotic/feeds on, Negative = competitive/parasitic
        self.ecological_relationships: Dict[int, float] = {}
        
        self.creation_time = time.time()
        self.age_generations = 0
        
        # Tracking for speciation (evolution vs replacement)
        self.original_genome_signature: Dict[int, float] = {}
        
        # History
        self.population_history: List[int] = []
        self.fitness_history: List[float] = []

    def update_genome_signature(self, recent_verbs: List[int]):
        """Update the dominant transformations used by this species."""
        if not recent_verbs:
            return
            
        # Calculate frequencies
        freqs = {}
        for v in recent_verbs:
            freqs[v] = freqs.get(v, 0) + 1.0
        total = sum(freqs.values())
        freqs = {k: v/total for k, v in freqs.items()}
        
        # Smooth update
        if not self.genome_signature:
            self.genome_signature = freqs
            self.original_genome_signature = freqs.copy()
        else:
            for k in set(self.genome_signature.keys()) | set(freqs.keys()):
                old_val = self.genome_signature.get(k, 0.0)
                new_val = freqs.get(k, 0.0)
                self.genome_signature[k] = 0.9 * old_val + 0.1 * new_val
                
    def check_speciation(self) -> bool:
        """
        Has this species drifted so far from its origin that it is a new species?
        Returns True if a speciation event should occur.
        """
        if not self.original_genome_signature or not self.genome_signature:
            return False
            
        # Calculate cosine distance between original and current verb usage
        all_keys = set(self.original_genome_signature.keys()) | set(self.genome_signature.keys())
        v1 = np.array([self.original_genome_signature.get(k, 0.0) for k in all_keys])
        v2 = np.array([self.genome_signature.get(k, 0.0) for k in all_keys])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return False
            
        similarity = np.dot(v1, v2) / (norm1 * norm2)
        distance = 1.0 - similarity
        
        # If verb usage has drifted by more than 40%, it's a new species
        return distance > 0.40

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tier": self.tier.value,
            "origin_node": self.origin_node,
            "members": list(self.members),
            "genome_signature": self.genome_signature,
            "original_genome_signature": self.original_genome_signature,
            "culture": self.culture.to_dict(),
            "ecological_relationships": self.ecological_relationships,
            "creation_time": self.creation_time,
            "age_generations": self.age_generations,
            "population_history": self.population_history,
            "fitness_history": self.fitness_history
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Species':
        s = cls(data["id"], data["origin_node"])
        s.tier = SpeciesTier(data["tier"])
        s.members = set(data["members"])
        
        # JSON converts int keys to strings, convert back
        s.genome_signature = {int(k): v for k, v in data.get("genome_signature", {}).items()}
        s.original_genome_signature = {int(k): v for k, v in data.get("original_genome_signature", {}).items()}
        s.ecological_relationships = {int(k): v for k, v in data.get("ecological_relationships", {}).items()}
        
        s.culture = Culture.from_dict(data.get("culture", {}))
        s.creation_time = data["creation_time"]
        s.age_generations = data.get("age_generations", 0)
        s.population_history = data.get("population_history", [])
        s.fitness_history = data.get("fitness_history", [])
        return s


class Ecologist:
    """
    Discovers and tracks species.
    Uses Louvain clustering as a detector, but enforces ancestry and verb constraints.
    """
    def __init__(self, db_conn, db_lock=None):
        self.db_conn = db_conn
        self.db_lock = db_lock
        self.species: Dict[int, Species] = {}
        self.legends: Dict[int, LegendNode] = {}
        self.node_to_species: Dict[int, int] = {} # concept_id -> species_id
        self.next_id = 1
        self.next_legend_id = 1
        self._init_db()
        self.load_from_db()

    def _init_db(self):
        if self.db_lock:
            with self.db_lock:
                self._init_db_impl()
        else:
            self._init_db_impl()
    
    def _init_db_impl(self):
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ecology_species (
                id INTEGER PRIMARY KEY,
                data JSON
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ecology_legends (
                id INTEGER PRIMARY KEY,
                data JSON
            )
        ''')
        self.db_conn.commit()

    def discover_species(self, concept_graph, physics_engine, current_age: int):
        """
        Run the 4-condition species detection.
        1. Graph Cohesion (Louvain)
        2. Shared Ancestry
        3. Shared Verbs
        """
        print("\n[Ecologist] Running species discovery sweep...")
        
        # 1. Graph Cohesion (Build undirected graph for Louvain)
        G = nx.Graph()
        from concept_graph import NodeTier
        for node_id, node in concept_graph.nodes.items():
            if node.tier == NodeTier.REALITY: continue # Reality nodes don't form species
            G.add_node(node_id)
            for parent_id in node.parents:
                if parent_id in concept_graph.nodes:
                    G.add_edge(parent_id, node_id, weight=1.0)
            for target_id, weight in node.visual_similar:
                G.add_edge(node_id, target_id, weight=weight * 0.5) # Ancestry matters more
            for target_id, weight in node.latent_similar:
                G.add_edge(node_id, target_id, weight=weight * 0.5)
            for target_id, weight in node.functional_similar:
                G.add_edge(node_id, target_id, weight=weight * 0.5)
                
        if len(G.nodes) < 20:
            return # Too small for ecology
            
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight='weight')
        except ImportError:
            print("[Ecologist] python-louvain not installed. Skipping cohesion check.")
            return
            
        # Group nodes by community
        communities = {}
        for node_id, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node_id)
            
        # Evaluate each community
        for comm_id, members in communities.items():
            if len(members) < 10:
                continue # Too small to be a species
                
            # 2. Shared Ancestry Check
            # Find the oldest node in the cluster
            oldest_node = min(members, key=lambda x: concept_graph.nodes[x].age if x in concept_graph.nodes else float('inf'))
            
            # Check if a majority of members trace back to this node (directed reachability check)
            reachable_count = 0
            for member_id in members:
                if member_id == oldest_node:
                    reachable_count += 1
                    continue
                # Simple BFS up the parent chain
                visited = set()
                queue = [member_id]
                found = False
                while queue:
                    curr = queue.pop(0)
                    if curr == oldest_node:
                        found = True
                        break
                    if curr not in visited and curr in concept_graph.nodes:
                        visited.add(curr)
                        queue.extend(concept_graph.nodes[curr].parents)
                if found:
                    reachable_count += 1
            
            # If less than 30% of members share this ancestry, it's not a cohesive species
            if reachable_count / len(members) < 0.3:
                continue
            
            # 3. Shared Verb Check
            # Collect verbs used to create these members
            verbs_used = []
            for member_id in members:
                # We need to know which verb created this member. 
                # For now, we approximate by looking at the physics engine's domains.
                for verb in physics_engine.transformations.values():
                    if member_id in verb.applicability_domain:
                        verbs_used.append(verb.id)
                        
            # If no verbs are shared, it's just a visual cluster, not a species
            if not verbs_used:
                continue
                
            # Is this an existing species?
            existing_species_id = None
            for member_id in members:
                if member_id in self.node_to_species:
                    existing_species_id = self.node_to_species[member_id]
                    break
                    
            if existing_species_id and existing_species_id in self.species:
                # Update existing species
                s = self.species[existing_species_id]
                s.members.update(members)
                for m in members:
                    self.node_to_species[m] = s.id
                s.update_genome_signature(verbs_used)
                
                # Check for Speciation (Evolution vs Replacement)
                if s.check_speciation():
                    print(f"[Ecologist] SPECIATION EVENT! Species {s.id} has drifted too far from its origin.")
                    s.tier = SpeciesTier.ANCIENT_SPECIES
                    
                    # Create new proto-species from current members
                    new_s = Species(self.next_id, oldest_node)
                    self.next_id += 1
                    new_s.members = s.members.copy()
                    new_s.update_genome_signature(verbs_used)
                    
                    # Inherit culture but with slight mutation
                    new_s.culture = Culture.from_dict(s.culture.to_dict())
                    new_s.culture.origin_legend = f"Descended from the Ancient {s.id}"
                    
                    self.species[new_s.id] = new_s
                    for m in new_s.members:
                        self.node_to_species[m] = new_s.id
                        
                    self.save_species(s)
                    self.save_species(new_s)
                else:
                    s.age_generations += 1
                    self.save_species(s)
                    
            else:
                # Create new Proto-Species
                s = Species(self.next_id, oldest_node)
                self.next_id += 1
                s.members = set(members)
                for m in members:
                    self.node_to_species[m] = s.id
                s.update_genome_signature(verbs_used)
                
                # Generate initial culture biases randomly
                s.culture.aesthetic_biases = {
                    "Beauty": np.random.uniform(0.8, 1.2),
                    "Novelty": np.random.uniform(0.8, 1.2),
                    "Coherence": np.random.uniform(0.8, 1.2)
                }
                
                self.species[s.id] = s
                self.save_species(s)
                print(f"[Ecologist] Discovered new Proto-Species {s.id} with {len(members)} members.")

    def run_historian_sweep(self, concept_graph, physics_engine):
        """
        Phase 6: Deep Time - The Historian.
        Evaluates extinct species. If they have high mythic weight, they become Legends.
        Otherwise, they are forgotten.
        """
        print("\n[Historian] Running historical compression sweep...")
        
        # 1. Identify extinct species (no living members in the active concept graph)
        active_nodes = set(concept_graph.nodes.keys())
        
        for s_id, s in list(self.species.items()):
            if s.tier in [SpeciesTier.EXTINCT, SpeciesTier.FORGOTTEN]:
                continue
                
            living_members = s.members.intersection(active_nodes)
            if not living_members and s.tier != SpeciesTier.ANCIENT_SPECIES:
                # Species has died out
                print(f"[Historian] Species {s.id} has gone extinct.")
                
                # 2. Myth Selection Pressure (Does it deserve to be a legend?)
                offspring_count = sum(1 for m in s.members if m in concept_graph.nodes and concept_graph.nodes[m].offspring_count > 0)
                transformation_influence = len(s.genome_signature) # How many verbs it mastered
                
                # Calculate legend score
                legend_score = (offspring_count * transformation_influence * (s.age_generations / 100.0))
                
                if legend_score > 50.0:
                    # 3. Create Legend Node (Memory Mutation)
                    print(f"[Historian] Species {s.id} achieved mythic status (Score: {legend_score:.1f}). Writing Legend.")
                    s.tier = SpeciesTier.EXTINCT
                    
                    legend = LegendNode(self.next_legend_id, s.id)
                    self.next_legend_id += 1
                    
                    # Compression Error: Historian loses details, keeps only top 2 verbs
                    sorted_verbs = sorted(s.genome_signature.items(), key=lambda x: x[1], reverse=True)
                    legend.dominant_verbs = [v[0] for v in sorted_verbs[:2]]
                    
                    # Narrative Bias: Assign a mythic name based on its culture/verbs
                    legend.remembered_name = f"The First Age of Species {s.id}"
                    if s.culture.origin_legend:
                        legend.famous_events.append(f"Origin: {s.culture.origin_legend}")
                        
                    # Symbolic Transformation: Average the latent vectors of its top 5 members to create an Archetype
                    top_members = sorted(list(s.members), key=lambda x: concept_graph.nodes[x].fitness if x in concept_graph.nodes else 0, reverse=True)[:5]
                    archetype = None
                    for m in top_members:
                        if m in concept_graph.nodes:
                            lat = concept_graph.nodes[m].latent
                            if lat is None:
                                # Fetch latent from database if not loaded
                                if self.db_lock:
                                    with self.db_lock:
                                        cursor = self.db_conn.cursor()
                                        cursor.execute("SELECT latent FROM concepts WHERE id = ?", (m,))
                                        row = cursor.fetchone()
                                else:
                                    cursor = self.db_conn.cursor()
                                    cursor.execute("SELECT latent FROM concepts WHERE id = ?", (m,))
                                    row = cursor.fetchone()
                                if row and row[0]:
                                    lat = np.array(json.loads(row[0]), dtype=np.float32)
                                    concept_graph.nodes[m].latent = lat
                                else:
                                    continue
                            
                            if archetype is None:
                                archetype = lat.copy()
                            else:
                                archetype += lat
                    if archetype is not None:
                        archetype /= len(top_members)
                        
                        # Add hallucination (error rate)
                        legend.accuracy = np.random.uniform(0.7, 0.95)
                        noise = np.random.randn(*archetype.shape) * (1.0 - legend.accuracy)
                        archetype += noise
                        legend.archetype_latent = archetype.tolist()
                        
                    legend.mythic_weight = legend_score
                    self.legends[legend.id] = legend
                    self.save_legend(legend)
                    
                else:
                    print(f"[Historian] Species {s.id} forgotten by time (Score: {legend_score:.1f}).")
                    s.tier = SpeciesTier.FORGOTTEN
                    
                self.save_species(s)

    def save_species(self, s: Species):
        if self.db_lock:
            with self.db_lock:
                self._save_species_impl(s)
        else:
            self._save_species_impl(s)
    
    def _save_species_impl(self, s: Species):
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO ecology_species (id, data)
            VALUES (?, ?)
        ''', (s.id, json.dumps(s.to_dict())))
        self.db_conn.commit()

    def save_legend(self, l: LegendNode):
        if self.db_lock:
            with self.db_lock:
                self._save_legend_impl(l)
        else:
            self._save_legend_impl(l)
    
    def _save_legend_impl(self, l: LegendNode):
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO ecology_legends (id, data)
            VALUES (?, ?)
        ''', (l.id, json.dumps(l.to_dict())))
        self.db_conn.commit()

    def load_from_db(self):
        if self.db_lock:
            with self.db_lock:
                self._load_from_db_impl()
        else:
            self._load_from_db_impl()
    
    def _load_from_db_impl(self):
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ecology_species'")
        if not cursor.fetchone():
            return

        cursor.execute("SELECT data FROM ecology_species")
        for row in cursor:
            data = json.loads(row[0])
            s = Species.from_dict(data)
            self.species[s.id] = s
            for m in s.members:
                self.node_to_species[m] = s.id
            if s.id >= self.next_id:
                self.next_id = s.id + 1
                
        # Load legends
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ecology_legends'")
        if cursor.fetchone():
            cursor.execute("SELECT data FROM ecology_legends")
            for row in cursor:
                data = json.loads(row[0])
                l = LegendNode.from_dict(data)
                self.legends[l.id] = l
                if l.id >= self.next_legend_id:
                    self.next_legend_id = l.id + 1
                
        print(f"Loaded {len(self.species)} species and {len(self.legends)} legends from ecology engine.")
