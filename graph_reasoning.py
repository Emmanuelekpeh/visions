"""
Graph Neural Network Reasoning Layer
Sits above the VAE to learn patterns from the concept graph.

Purpose: Recommendation engine for imagination.
Learns:
- "Concepts near this region usually produce good offspring"
- "forest + glass → interesting offspring"
- "metal forest is unexplored but similar to successful combinations"

Not a replacement for the VAE - it's a meta-learner that guides exploration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from concept_graph import ConceptGraph, ConceptNode, EdgeType, NodeTier
from training_scheduler import run_on_pytorch_thread

class GraphConvLayer(nn.Module):
    """
    Simple graph convolution layer.
    Aggregates information from neighbors weighted by edge type.
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_edge_types: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Learnable weights for each edge type
        self.edge_weights = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_edge_types)
        ])
        
        # Self-loop
        self.self_weight = nn.Linear(in_dim, out_dim)
        
    def forward(self, node_features: torch.Tensor, 
                adj_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            node_features: [N, in_dim] - features for N nodes
            adj_matrices: List of [N, N] adjacency matrices (one per edge type)
        
        Returns:
            new_features: [N, out_dim]
        """
        # Self-loop
        out = self.self_weight(node_features)
        
        # Aggregate from neighbors for each edge type
        for edge_type_idx, adj in enumerate(adj_matrices):
            # adj @ node_features gives sum of neighbor features
            neighbor_features = torch.mm(adj, node_features)
            out = out + self.edge_weights[edge_type_idx](neighbor_features)
            
        return F.relu(out)


class ConceptGNN(nn.Module):
    """
    Graph Neural Network for reasoning over concept graph.
    
    Predicts:
    1. Offspring quality: "Will this concept produce good children?"
    2. Combination success: "Will parent_a + parent_b → interesting offspring?"
    3. Exploration value: "Is this unexplored region promising?"
    """
    
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, num_edge_types=6)
            for _ in range(num_layers)
        ])
        
        # Prediction heads (predict both mean and uncertainty!)
        self.offspring_quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # [mean, log_variance]
        )
        
        self.combination_success_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # [mean, log_variance]
        )
        
        self.exploration_value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features: torch.Tensor, 
                adj_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Run graph convolutions to get node representations.
        
        Args:
            node_features: [N, embedding_dim]
            adj_matrices: List of [N, N] adjacency matrices
            
        Returns:
            node_representations: [N, hidden_dim]
        """
        # Project input
        x = self.input_proj(node_features)
        
        # Graph convolutions
        for conv in self.conv_layers:
            x = conv(x, adj_matrices)
            
        return x
        
    def predict_offspring_quality(self, node_representations: torch.Tensor, 
                                   node_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict: "Will this concept produce good offspring?"
        
        Returns: 
            means: [len(node_indices)] predicted quality in [0, 1]
            uncertainties: [len(node_indices)] prediction uncertainty in [0, 1]
        """
        node_feats = node_representations[node_indices]
        predictions = self.offspring_quality_head(node_feats)  # [N, 2]
        
        means = torch.sigmoid(predictions[:, 0])  # [0, 1]
        log_vars = predictions[:, 1]
        uncertainties = torch.sigmoid(log_vars)  # [0, 1]
        
        return means, uncertainties
        
    def predict_combination_success(self, node_representations: torch.Tensor,
                                     parent_a_indices: List[int],
                                     parent_b_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict: "Will parent_a + parent_b → good offspring?"
        
        Returns:
            means: [len(pairs)] predicted success in [0, 1]
            uncertainties: [len(pairs)] prediction uncertainty in [0, 1]
        """
        parent_a_feats = node_representations[parent_a_indices]
        parent_b_feats = node_representations[parent_b_indices]
        
        # Concatenate parent features
        combined = torch.cat([parent_a_feats, parent_b_feats], dim=-1)
        
        predictions = self.combination_success_head(combined)  # [N, 2]
        
        means = torch.sigmoid(predictions[:, 0])  # [0, 1]
        log_vars = predictions[:, 1]
        uncertainties = torch.sigmoid(log_vars)  # [0, 1]
        
        return means, uncertainties
        
    def predict_exploration_value(self, node_representations: torch.Tensor,
                                   node_indices: List[int]) -> torch.Tensor:
        """
        Predict: "Is this concept in an interesting unexplored region?"
        
        Returns: [len(node_indices)] scores in [0, 1]
        """
        node_feats = node_representations[node_indices]
        return self.exploration_value_head(node_feats).squeeze(-1)


class GraphReasoningSystem:
    """
    Wrapper around ConceptGNN for integration with the universe.
    
    This is the "recommendation engine for imagination".
    """
    
    def __init__(self, graph: ConceptGraph, embedding_dim: int = 512):
        self.graph = graph
        self.gnn = ConceptGNN(embedding_dim=embedding_dim)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.001)
        
        # Load weights if available
        import os
        self.weights_path = "gnn_weights.pt"
        if os.path.exists(self.weights_path):
            try:
                self.gnn.load_state_dict(torch.load(self.weights_path, weights_only=True))
                print("  [GNN] Loaded trained weights from disk.")
            except Exception as e:
                print(f"  [GNN] Could not load weights: {e}")
                
        # Training data buffer
        self.training_buffer = {
            "offspring_quality": [],  # (node_id, actual_quality)
            "combination_success": [],  # (parent_a, parent_b, success_score)
        }
        
    def _build_adjacency_matrices(self, max_nodes: int = None) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[int, int]]:
        """
        Build adjacency matrices for each edge type.

        Args:
            max_nodes: If set and graph is larger, subsample nodes (prefer recent /
                       high-fitness / reality) so dense N×N tensors stay CPU-safe.
        
        Returns:
            node_features: [N, embedding_dim]
            adj_matrices: List of [N, N] matrices
            id_to_idx: Mapping from node_id to matrix index
        """
        N_full = len(self.graph.nodes)
        if N_full == 0:
            return torch.zeros((0, 512)), [], {}
            
        # Create snapshot of node IDs to avoid race conditions
        node_ids = sorted(list(self.graph.nodes.keys()))

        if max_nodes is not None and len(node_ids) > max_nodes:
            # Prefer reality anchors, then high fitness, then recent age
            scored = []
            for nid in node_ids:
                node = self.graph.nodes[nid]
                tier_bonus = 10.0 if node.tier == NodeTier.REALITY else 0.0
                scored.append((tier_bonus + float(node.fitness) + 0.001 * float(node.age), nid))
            scored.sort(reverse=True)
            # Always include nodes that appear in the training buffer
            must_keep = set()
            for nid, _, _ in self.training_buffer.get("offspring_quality", []):
                must_keep.add(nid)
            for a, b, _, _ in self.training_buffer.get("combination_success", []):
                must_keep.add(a)
                must_keep.add(b)
            selected = [nid for nid in must_keep if nid in self.graph.nodes]
            for _, nid in scored:
                if nid not in must_keep:
                    selected.append(nid)
                if len(selected) >= max_nodes:
                    break
            node_ids = sorted(selected[:max_nodes])

        N = len(node_ids)
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        # Extract node features (embeddings)
        first_node = self.graph.nodes[node_ids[0]]
        embedding_dim = first_node.embedding.flatten().shape[0]
        
        node_features = np.zeros((N, embedding_dim), dtype=np.float32)
        for idx, nid in enumerate(node_ids):
            embedding = self.graph.nodes[nid].embedding.flatten()
            node_features[idx] = embedding
            
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Build adjacency matrices for each edge type
        edge_types = [
            EdgeType.EVOLVED_FROM,
            EdgeType.VISUAL_SIMILAR,
            EdgeType.COMBINES_WITH,
            EdgeType.CONTRADICTS,
            EdgeType.SPECIALIZES_INTO,
            EdgeType.FAILED_INVALID  # Represents all hazard edges combined
        ]
        
        adj_matrices = []
        
        for edge_type in edge_types:
            adj = torch.zeros((N, N), dtype=torch.float32)
            
            for nid in node_ids:
                node = self.graph.nodes[nid]
                src_idx = id_to_idx[nid]
                
                if edge_type == EdgeType.EVOLVED_FROM:
                    for parent_id in node.parents:
                        if parent_id in id_to_idx:
                            tgt_idx = id_to_idx[parent_id]
                            adj[src_idx, tgt_idx] = 1.0
                            
                elif edge_type == EdgeType.VISUAL_SIMILAR:
                    for other_id, sim in node.visual_similar:
                        if other_id in id_to_idx:
                            tgt_idx = id_to_idx[other_id]
                            adj[src_idx, tgt_idx] = max(float(adj[src_idx, tgt_idx]), float(sim))
                    for other_id, sim in node.latent_similar:
                        if other_id in id_to_idx:
                            tgt_idx = id_to_idx[other_id]
                            adj[src_idx, tgt_idx] = max(float(adj[src_idx, tgt_idx]), float(sim))
                    for other_id, sim in node.functional_similar:
                        if other_id in id_to_idx:
                            tgt_idx = id_to_idx[other_id]
                            adj[src_idx, tgt_idx] = max(float(adj[src_idx, tgt_idx]), float(sim))
                            
                elif edge_type == EdgeType.CONTRADICTS:
                    for other_id in node.contradicts:
                        if other_id in id_to_idx:
                            tgt_idx = id_to_idx[other_id]
                            adj[src_idx, tgt_idx] = 1.0
                            
                elif edge_type == EdgeType.SPECIALIZES_INTO:
                    for other_id in node.specializes_into:
                        if other_id in id_to_idx:
                            tgt_idx = id_to_idx[other_id]
                            adj[src_idx, tgt_idx] = 1.0
                            
                # Hazard Map Edges (New 6th adjacency matrix)
                elif edge_type == EdgeType.FAILED_INVALID:
                    # Treat all hazard types as a single negative adjacency for the GNN
                    for other_id in node.failed_invalid + node.failed_unstable + node.failed_sterile + node.failed_catastrophic:
                        if other_id in id_to_idx:
                            tgt_idx = id_to_idx[other_id]
                            adj[src_idx, tgt_idx] = 1.0
                            
            adj_matrices.append(adj)
            
        return node_features, adj_matrices, id_to_idx
        
    def recommend_parents_for_mutation(self, k: int = 5, current_time: float = 0.0) -> List[Tuple[int, float, float, float]]:
        """
        Recommend k concepts that are likely to produce good offspring.
        
        Combines GNN prediction with exploration bonus to prevent getting stuck.
        
        Returns: List of (node_id, predicted_quality, uncertainty, exploration_bonus)
        """
        if len(self.graph.nodes) == 0:
            return []
            
        return run_on_pytorch_thread(self._recommend_parents_for_mutation_impl, k, current_time)

    def _recommend_parents_for_mutation_impl(self, k: int, current_time: float) -> List[Tuple[int, float, float, float]]:
        node_features, adj_matrices, id_to_idx = self._build_adjacency_matrices(max_nodes=256)
        if len(node_features) == 0:
            return []

        with torch.no_grad():
            node_reps = self.gnn(node_features, adj_matrices)
            all_indices = list(range(len(node_features)))
            qualities, uncertainties = self.gnn.predict_offspring_quality(node_reps, all_indices)
            qualities = qualities.detach().cpu().tolist()
            uncertainties = uncertainties.detach().cpu().tolist()
            
            exploration_vals = self.gnn.predict_exploration_value(node_reps, all_indices)
            exploration_vals = exploration_vals.detach().cpu().tolist()
            
        # Map back to node IDs and compute exploration bonuses
        # Snapshot to avoid race conditions
        node_ids = sorted(list(id_to_idx.keys()), key=lambda nid: id_to_idx[nid])
        recommendations = []
        
        for idx in range(len(node_ids)):
            if idx >= len(qualities) or idx >= len(exploration_vals):
                break # safeguard against dimension mismatch during active async ingestion
            node_id = node_ids[idx]
            
            # Safety check: node might have been removed during evolution
            if node_id not in self.graph.nodes:
                continue
                
            node = self.graph.nodes[node_id]
            
            quality = float(qualities[idx])
            uncertainty = float(uncertainties[idx])
            exploration = float(exploration_vals[idx])
            
            # Combined score:
            # high reward + low uncertainty = exploit
            # high reward + high uncertainty = investigate
            # low reward + high exploration = explore blank spaces
            score = (
                0.40 * quality +
                0.25 * (quality * uncertainty) +  # Investigate uncertain high-reward regions
                0.20 * exploration +
                0.15 * (1.0 - node.confidence)  # Give low-confidence nodes a chance
            )
            
            recommendations.append((node_id, score, uncertainty, exploration))
            
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:k]
        
    def recommend_parent_pairs_for_recombination(self, k: int = 5, current_time: float = 0.0) -> List[Tuple[int, int, float, float]]:
        """
        Recommend k pairs of concepts that are likely to combine well.
        
        Returns: List of (parent_a_id, parent_b_id, predicted_success, uncertainty)
        """
        if len(self.graph.nodes) < 2:
            return []
            
        return run_on_pytorch_thread(self._recommend_parent_pairs_impl, k, current_time)

    def _recommend_parent_pairs_impl(self, k: int, current_time: float) -> List[Tuple[int, int, float, float]]:
        node_features, adj_matrices, id_to_idx = self._build_adjacency_matrices(max_nodes=256)
        if len(node_features) == 0:
            return []

        node_ids = list(id_to_idx.keys())

        import random
        candidate_pairs = []
        for _ in range(min(100, max(1, len(node_ids) * len(node_ids) // 10))):
            if len(node_ids) < 2:
                break
            a, b = random.sample(node_ids, 2)
            if a in id_to_idx and b in id_to_idx:
                candidate_pairs.append((a, b))

        if not candidate_pairs:
            return []

        with torch.no_grad():
            node_reps = self.gnn(node_features, adj_matrices)
            parent_a_indices = [id_to_idx[a] for a, b in candidate_pairs]
            parent_b_indices = [id_to_idx[b] for a, b in candidate_pairs]
            success_scores, uncertainties = self.gnn.predict_combination_success(
                node_reps, parent_a_indices, parent_b_indices
            )
            success_scores = success_scores.detach().cpu().tolist()
            uncertainties = uncertainties.detach().cpu().tolist()
            
            all_indices = list(range(len(node_features)))
            exploration_vals = self.gnn.predict_exploration_value(node_reps, all_indices)
            exploration_vals = exploration_vals.detach().cpu().tolist()
            
        # Combine results with exploration bonus
        recommendations = []
        for i, (a, b) in enumerate(candidate_pairs):
            # Safety check: ensure both nodes still exist
            if a not in self.graph.nodes or b not in self.graph.nodes:
                continue
                
            success = float(success_scores[i])
            uncertainty = float(uncertainties[i])
            
            # Exploration bonuses for both parents
            exploration_a = float(exploration_vals[id_to_idx[a]]) if a in id_to_idx else 0.0
            exploration_b = float(exploration_vals[id_to_idx[b]]) if b in id_to_idx else 0.0
            avg_exploration = (exploration_a + exploration_b) / 2.0
            
            # Combined score
            score = (
                0.50 * success +
                0.20 * (success * uncertainty) +  # Investigate uncertain high-success pairs
                0.30 * avg_exploration
            )
            
            recommendations.append((a, b, score, uncertainty))
            
        recommendations.sort(key=lambda x: x[2], reverse=True)
        
        return recommendations[:k]
        
    def get_exploration_values(self, node_ids: List[int]) -> Dict[int, float]:
        """Get GNN predicted exploration values for specific nodes."""
        if not node_ids:
            return {}
        return run_on_pytorch_thread(self._get_exploration_values_impl, node_ids)

    def _get_exploration_values_impl(self, node_ids: List[int]) -> Dict[int, float]:
        node_features, adj_matrices, id_to_idx = self._build_adjacency_matrices(max_nodes=256)
        if len(node_features) == 0:
            return {nid: 0.0 for nid in node_ids}
            
        with torch.no_grad():
            node_reps = self.gnn(node_features, adj_matrices)
            all_indices = list(range(len(node_features)))
            exploration_values = self.gnn.predict_exploration_value(node_reps, all_indices)
            exploration_values = exploration_values.detach().cpu().tolist()
            
        result = {}
        for nid in node_ids:
            if nid in id_to_idx:
                idx = id_to_idx[nid]
                if idx < len(exploration_values):
                    result[nid] = float(exploration_values[idx])
                else:
                    result[nid] = 0.0
            else:
                result[nid] = 0.0
                
        return result

    def train_step(self, max_nodes: int = 256):
        """
        Train the GNN on accumulated data.
        Learns from past successes/failures.

        CPU-aware: if the graph is large, subsample nodes so dense N×N
        adjacency build + backward cannot OOM-kill the process (often around
        the gen-100 train/save tick).
        """
        if len(self.training_buffer["offspring_quality"]) < 10:
            return 0.0  # Not enough data yet

        return run_on_pytorch_thread(self._train_step_impl, max_nodes)

    def _train_step_impl(self, max_nodes: int = 256):
        try:
            node_features, adj_matrices, id_to_idx = self._build_adjacency_matrices(
                max_nodes=max_nodes
            )

            if len(node_features) == 0:
                return 0.0

            node_reps = self.gnn(node_features, adj_matrices)

            # Loss 1: Offspring quality prediction
            if self.training_buffer["offspring_quality"]:
                node_ids = [nid for nid, _, _ in self.training_buffer["offspring_quality"]]
                actual_qualities = [q for _, q, _ in self.training_buffer["offspring_quality"]]
                timestamps = [t for _, _, t in self.training_buffer["offspring_quality"]]

                valid_pairs = [
                    (id_to_idx[nid], q, t)
                    for nid, q, t in zip(node_ids, actual_qualities, timestamps)
                    if nid in id_to_idx
                ]

                if valid_pairs:
                    indices = [idx for idx, _, _ in valid_pairs]
                    targets = torch.tensor([q for _, q, _ in valid_pairs], dtype=torch.float32)
                    max_t = max(timestamps) if timestamps else 0
                    time_weights = torch.tensor(
                        [np.exp(-(max_t - t) / 1000.0) for _, _, t in valid_pairs],
                        dtype=torch.float32,
                    )
                    predictions, _ = self.gnn.predict_offspring_quality(node_reps, indices)
                    loss_offspring = torch.mean(time_weights * (predictions - targets) ** 2)
                else:
                    loss_offspring = torch.tensor(0.0)
            else:
                loss_offspring = torch.tensor(0.0)

            # Loss 2: Combination success prediction
            if self.training_buffer["combination_success"]:
                parent_a_ids = [a for a, b, s, t in self.training_buffer["combination_success"]]
                parent_b_ids = [b for a, b, s, t in self.training_buffer["combination_success"]]
                success_scores = [s for a, b, s, t in self.training_buffer["combination_success"]]
                timestamps = [t for a, b, s, t in self.training_buffer["combination_success"]]

                valid_triplets = [
                    (id_to_idx[a], id_to_idx[b], s, t)
                    for a, b, s, t in zip(parent_a_ids, parent_b_ids, success_scores, timestamps)
                    if a in id_to_idx and b in id_to_idx
                ]

                if valid_triplets:
                    a_indices = [a for a, b, s, t in valid_triplets]
                    b_indices = [b for a, b, s, t in valid_triplets]
                    targets = torch.tensor([s for a, b, s, t in valid_triplets], dtype=torch.float32)
                    max_t = max(timestamps) if timestamps else 0
                    time_weights = torch.tensor(
                        [np.exp(-(max_t - t) / 1000.0) for _, _, _, t in valid_triplets],
                        dtype=torch.float32,
                    )
                    predictions, _ = self.gnn.predict_combination_success(
                        node_reps, a_indices, b_indices
                    )
                    loss_combination = torch.mean(time_weights * (predictions - targets) ** 2)
                else:
                    loss_combination = torch.tensor(0.0)
            else:
                loss_combination = torch.tensor(0.0)

            total_loss = loss_offspring + loss_combination

            if total_loss.item() > 0:
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), 1.0)
                self.optimizer.step()
                torch.save(self.gnn.state_dict(), self.weights_path)

                self.training_buffer["offspring_quality"] = []
                self.training_buffer["combination_success"] = []

                loss_value = float(total_loss.detach().item())
                del node_features, adj_matrices, node_reps, total_loss
                return loss_value
            return 0.0
        except Exception as e:
            # Never let GNN training take down the universe process
            print(f"[GNN] train_step failed (continuing evolution): {e}")
            return 0.0
            
    def record_offspring_quality(self, parent_id: int, actual_fitness: float):
        """Record that parent_id produced offspring with actual_fitness."""
        import time
        self.training_buffer["offspring_quality"].append((parent_id, actual_fitness, time.time()))
        
    def record_combination_success(self, parent_a: int, parent_b: int, success_score: float):
        """Record that parent_a + parent_b → offspring with success_score."""
        import time
        self.training_buffer["combination_success"].append((parent_a, parent_b, success_score, time.time()))

    def record_hazard(self, parent_id: int, failure_type: str):
        """
        Phase 4: Record a toxic/forbidden region (hazard map).
        """
        import time
        # Assign a severe negative score based on failure type
        penalty = {
            "failed_invalid": -0.5,
            "failed_unstable": -0.8,
            "failed_sterile": -0.3,
            "failed_catastrophic": -1.0
        }.get(failure_type, -0.5)
        
        self.training_buffer["offspring_quality"].append((parent_id, penalty, time.time()))


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("GRAPH REASONING SYSTEM (GNN)")
    print("=" * 60)
    
    from concept_graph import ConceptGraph, ConceptNode, NodeTier
    
    # Create test graph
    graph = ConceptGraph()
    
    print("\nCreating test graph...")
    for i in range(20):
        node = ConceptNode(
            id=i,
            embedding=np.random.randn(512),
            tier=NodeTier.REALITY if i < 5 else NodeTier.DREAM,
            confidence=1.0 if i < 5 else 0.5,
            age=i,
            fitness=0.8 if i < 5 else 0.5,
            latent=np.random.randn(4, 64, 64)
        )
        graph.add_node(node)
        
    print(f"  Created {len(graph.nodes)} nodes")
    
    # Create GNN
    print("\nInitializing GNN...")
    gnn_system = GraphReasoningSystem(graph, embedding_dim=512)
    print("  [OK] GNN ready")
    
    # Test recommendations
    print("\nTesting offspring quality recommendations...")
    recommendations = gnn_system.recommend_parents_for_mutation(k=5)
    print(f"  Top 5 parents for mutation:")
    for node_id, quality in recommendations[:5]:
        print(f"    Node {node_id}: predicted quality = {quality:.3f}")
        
    print("\nTesting combination recommendations...")
    pair_recommendations = gnn_system.recommend_parent_pairs_for_recombination(k=3)
    print(f"  Top 3 parent pairs for recombination:")
    for a, b, success in pair_recommendations[:3]:
        print(f"    ({a}, {b}): predicted success = {success:.3f}")
        
    print("\n[Graph Reasoning System Ready]")
    print("This GNN learns: 'Which concepts produce good offspring?'")
    print("                 'Which combinations are promising?'")
