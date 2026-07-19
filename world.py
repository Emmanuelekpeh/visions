import time
import torch
import numpy as np
import json
import random
from database import WorldMemory
from engines import DreamEngine, CuriosityEngine
from vision import VisionSystem
from multihead_rl import create_multihead_system
from epistemic_memory import EpistemicMemory, MemoryTier
from meta_controller import MetaController, SystemState
from concept_graph import ConceptGraph, ConceptNode, NodeTier, EdgeType
from graph_reasoning import GraphReasoningSystem
from image_logger import ImageLogger
from physics import PhysicsEngine, FailureType
from ecology import Ecologist, SpeciesTier
from latent_generator import create_latent_generator
from training_scheduler import get_training_scheduler
import threading

class WorldState:
    def __init__(self, device="cpu"):
        self.device = device
        self.lock = threading.RLock()
        self.memory = WorldMemory()
        self.vision = VisionSystem(device=device)
        self.dreamer = DreamEngine(device=device)
        self.curiosity = CuriosityEngine()
        self.multihead_rl = create_multihead_system()
        
        # PHASE 1: Add concept graph as memory layer
        # Graph sits BELOW epistemic memory and feeds evidence UP
        print("Initializing concept graph...")
        self.concept_graph = ConceptGraph(similarity_threshold=0.85)
        self.concept_graph.load_from_database(self.memory.conn)
        
        # Graph Neural Network for reasoning (recommendation engine)
        print("Initializing GNN reasoning system...")
        self.gnn = GraphReasoningSystem(self.concept_graph, embedding_dim=512)
        
        # Epistemic memory (sits ABOVE graph, receives evidence from graph)
        print("Initializing epistemic memory...")
        self.epistemic_memory = EpistemicMemory(self.vision, self.memory.index)
        self.epistemic_memory.load_from_database(self.memory.conn)
        
        # Physics Engine (Phase 1: Transformation Objects)
        print("Initializing physics engine...")
        self.physics_engine = PhysicsEngine(self.memory.conn)
        
        # Ecology Engine (Phase 6: Species Discovery)
        print("Initializing ecology engine...")
        self.ecologist = Ecologist(self.memory.conn)
        
        # Latent Generator (solves unbounded storage growth)
        print("Initializing latent generator...")
        self.latent_generator, self.latent_generator_trainer = create_latent_generator(
            self.vision, device=device
        )
        
        # Meta-controller for adaptive RL head weighting
        self.meta_controller = MetaController()
        
        # Prediction tracking
        self.expected_scores = {}  # concept_id -> expected_score
        
        # Phase 6: Global Resource Economy (Scarcity)
        self.global_energy_pool = 1000.0  # The universe's total available energy
        self.max_energy_pool = 5000.0
        self.energy_replenishment_rate = 25.0  # Was 10 — birth costs were starving the pool
        
        # GNN influence ratio (meta-controller will adjust this)
        self.gnn_influence = 0.30  # Start at 30% GNN, 70% exploration (reduced to break stagnation)
        
        # Image logger for sparse output saving
        self.image_logger = ImageLogger(base_dir="outputs", save_interval=25)
        
        self.age = 0
        self.active_concepts = []
        self.current_latent = None
        self.current_image = None
        # Ghost vectors left in IndexFlatL2 after kills; rebuild mid-run when this rises
        self.faiss_ghosts = 0
        self.faiss_rebuild_threshold = 25
        
        # Kill ghost IDs before evolution starts (restart crash root cause)
        self._resync_living_memory()
        self._initialize_world()
        self._ensure_valid_active_concepts()

    def _persist_memory_layers(self, reason: str = "manual"):
        """Single persist path for graph + epistemic + FAISS (avoid double-writes)."""
        print(f"[Memory] Saving state ({reason})...")
        self.concept_graph.save_to_database(self.memory.conn)
        self.epistemic_memory.save_to_database(self.memory.conn)
        self.memory.save_faiss_index()
        print(f"[Memory] Save complete ({reason})")

    def get_latent_for_concept(self, concept_id: int):
        """
        Get latent for a concept - either from storage (real images) or generated (evolved concepts).
        Returns None if the concept no longer exists or latent cannot be recovered.
        Never raises — extinct/ghost IDs must not kill the process.
        """
        try:
            concept_id = int(concept_id)
        except (TypeError, ValueError):
            return None

        if not self.memory.concept_exists(concept_id):
            return None

        try:
            cursor = self.memory.conn.cursor()
            cursor.execute(
                "SELECT filename FROM ingested_files WHERE concept_id = ?",
                (concept_id,),
            )
            is_real_image = cursor.fetchone() is not None

            row = self.memory.get_concept(concept_id)
            if not row:
                return None

            # Prefer stored latent when present (real images / legacy)
            if row[2]:
                latent_np = np.array(json.loads(row[2]), dtype=np.float32)
                latent = torch.tensor(latent_np, dtype=torch.float32).to(self.device)
                if latent.dim() == 3:
                    latent = latent.unsqueeze(0)
                return latent

            # Evolved / missing latent: generate from embedding
            if not row[1]:
                return None
            embedding_np = np.array(json.loads(row[1]), dtype=np.float32).flatten()
            embedding_t = torch.tensor(embedding_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.latent_generator.eval()
            with torch.no_grad():
                return self.latent_generator(embedding_t)
        except Exception as e:
            print(f"[Latent] Failed for concept {concept_id}: {e}")
            return None

    def _ensure_valid_active_concepts(self):
        """Drop extinct IDs from active set; fall back to any living concept."""
        self.active_concepts = [
            cid for cid in self.active_concepts if self.memory.concept_exists(cid)
        ]
        if self.active_concepts:
            return True

        living = list(self.memory.get_living_ids())
        if not living:
            return False

        # Prefer a reality anchor if available
        for rid in list(self.concept_graph.reality_nodes):
            if rid in living:
                self.active_concepts = [rid]
                break
        if not self.active_concepts:
            self.active_concepts = [living[0]]

        cid = self.active_concepts[0]
        lat = self.get_latent_for_concept(cid)
        if lat is None:
            return False
        self.current_latent = lat
        self.current_image = self.vision.decode_latent_to_image(lat)
        return True

    def _resync_living_memory(self):
        """
        After load: align FAISS, graph, and epistemic layers with the concepts table.
        This is the main fix for restart crashes from extinct / ghost concept IDs.
        """
        print("[Resync] Aligning memory layers with living concepts...")
        living = self.memory.get_living_ids()
        print(f"  Living concepts in DB: {len(living)}")

        # Orphan metadata rows
        self.memory.purge_orphan_metadata(living)

        # Drop graph nodes that aren't in DB
        dead_graph = [nid for nid in list(self.concept_graph.nodes.keys()) if nid not in living]
        for nid in dead_graph:
            self.concept_graph.quarantine_node(nid)
        if dead_graph:
            print(f"  Removed {len(dead_graph)} dead graph nodes.")

        # Drop epistemic entries that aren't in DB
        dead_ep = 0
        for store_name in ("landmark_concepts", "working_concepts", "frontier_concepts"):
            store = getattr(self.epistemic_memory, store_name, {})
            for cid in list(store.keys()):
                if cid not in living:
                    del store[cid]
                    dead_ep += 1
        if dead_ep:
            print(f"  Removed {dead_ep} dead epistemic entries.")

        # Clean parent links pointing at ghosts (in-memory graph)
        for nid, node in self.concept_graph.nodes.items():
            if node.parents:
                node.parents = [p for p in node.parents if p in living]

        # Clean dead parent_ids stored in SQLite (38+ orphans seen in audits)
        cursor = self.memory.conn.cursor()
        fixed_parents = 0
        for cid, pjson in cursor.execute("SELECT id, parent_ids FROM concepts").fetchall():
            if not pjson:
                continue
            try:
                parents = json.loads(pjson)
            except Exception:
                continue
            cleaned = [int(p) for p in parents if int(p) in living]
            if cleaned != list(parents):
                cursor.execute(
                    "UPDATE concepts SET parent_ids = ? WHERE id = ?",
                    (json.dumps(cleaned), cid),
                )
                fixed_parents += 1
        if fixed_parents:
            self.memory.conn.commit()
            print(f"  Cleaned dead parent_ids on {fixed_parents} concepts.")

        print(f"[Resync] Graph={len(self.concept_graph.nodes)} FAISS={self.memory.index.ntotal}")
    
    def _initialize_world(self):
        """Seed the world if empty or unanchored."""
        import os
        
        # Check if we have real-world anchor concepts
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ingested_files")
        ingested_count = cursor.fetchone()[0]
        
        # If no real images ingested yet, ingest dataset first (limit to 200 initially)
        if ingested_count == 0 and os.path.exists("dataset"):
            print("=" * 80)
            print("UNIVERSE INITIALIZATION: Anchoring in Reality")
            print("=" * 80)
            print("No anchor concepts found. Ingesting first 200 RANDOM dataset images to ground the universe...")
            results = self.scan_dataset(limit=200, randomize=True)
            success_count = sum(1 for _, success in results if success)
            print(f"\n[OK] Anchored universe with {success_count} real-world concepts")
            print("Remaining images will be ingested in the background.")
            print("=" * 80)
            
        # Now initialize current state
        # Create fresh cursor after ingestion to see newly inserted data
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT id, latent FROM concepts ORDER BY id DESC LIMIT 1")
        latest = cursor.fetchone()
        
        if not latest:
            print("Empty universe after ingestion attempt. Generating genesis spark using trained Latent Generator...")
            # Create a random semantic embedding and let the trained generator hallucinate the latent
            random_embedding = torch.randn((1, 512)).to(self.device)
            random_embedding = torch.nn.functional.normalize(random_embedding, p=2, dim=1)
            self.latent_generator.eval()
            with torch.no_grad():
                genesis_latent = self.latent_generator(random_embedding)
            
            self._process_new_concept(genesis_latent, parent_ids=[], generation=0)
        else:
            cursor.execute("SELECT COUNT(*) FROM concepts")
            count = cursor.fetchone()[0]
            print(f"Loaded universe with {count} concepts.")
            # Prefer a concept that still exists and can yield a latent
            latest_id = latest[0]
            lat = self.get_latent_for_concept(latest_id)
            if lat is None:
                living = list(self.memory.get_living_ids())
                if not living:
                    print("No living concepts with usable latents; genesis spark via trained Latent Generator...")
                    random_embedding = torch.randn((1, 512)).to(self.device)
                    random_embedding = torch.nn.functional.normalize(random_embedding, p=2, dim=1)
                    self.latent_generator.eval()
                    with torch.no_grad():
                        genesis_latent = self.latent_generator(random_embedding)
                    
                    self._process_new_concept(genesis_latent, parent_ids=[], generation=0)
                    return
                latest_id = living[-1]
                lat = self.get_latent_for_concept(latest_id)
            self.current_latent = lat
            if self.current_latent is not None:
                self.current_image = self.vision.decode_latent_to_image(self.current_latent)
            else:
                self.current_image = None
            self.active_concepts = [latest_id]
            self.age = self.memory.get_max_generation()

    def _process_new_concept(self, latent: torch.Tensor, parent_ids: list, generation: int, is_real_image: bool = False):
        """
        Process new concept with epistemic validation and ecological scarcity.
        """
        # Drop extinct parent refs immediately — never look up ghosts downstream
        parent_ids = [int(p) for p in (parent_ids or []) if self.memory.concept_exists(int(p))]
        # Extract concept embedding
        embedding_tensor = self.vision.extract_concept_from_latent(latent)
        embedding_np = embedding_tensor.cpu().numpy()
        
        # Phase 6: Scarcity - Calculate complexity to determine energy cost
        # Complexity must stay in a sane band: raw embedding "entropy" without
        # normalizing to a distribution was ~50-80 and made every birth cost ~40+
        # while replenishment is only +10/gen → permanent cannibalism.
        latent_var = float(torch.var(latent))
        abs_emb = np.abs(embedding_np.flatten()) + 1e-8
        abs_emb = abs_emb / abs_emb.sum()
        emb_entropy = -float(np.sum(abs_emb * np.log(abs_emb)))
        # Normalize entropy against log(dim) so it's ~[0,1]
        emb_entropy_norm = emb_entropy / max(1.0, np.log(len(abs_emb)))
        complexity = float(np.clip((latent_var + emb_entropy_norm) / 2.0, 0.5, 6.0))
        
        # Create ecology profile
        from concept_graph import ConceptEcology
        ecology = ConceptEcology(complexity)
        
        # Phase 6: Scarcity - Can the universe afford this concept?
        if not is_real_image and self.global_energy_pool < ecology.energy_cost:
            # The universe is starving. We must trigger a competition event.
            print(f"\n[Scarcity] Universe energy low ({self.global_energy_pool:.1f}). Triggering competition event for new concept...")
            if not self._run_competition_event(ecology.energy_cost):
                # Competition failed. The new concept dies before birth.
                return False, {"error": "Failed to secure ecological resources."}
        
        # Deduct energy for birth
        if not is_real_image:
            self.global_energy_pool -= ecology.energy_cost
        
        # Get parent embeddings and confidences
        parent_embs = []
        parent_confidences = []
        for pid in parent_ids:
            row = self.memory.get_concept(pid)
            if row:
                parent_embs.append(np.array(json.loads(row[1]), dtype=np.float32))
                # Get confidence from epistemic memory
                conf_obj = self.epistemic_memory._get_concept(pid)
                if conf_obj:
                    parent_confidences.append(conf_obj.confidence)
                else:
                    parent_confidences.append(0.5)  # Unknown parent
                    
        # Predict expected score (for prediction error tracking)
        if parent_confidences:
            expected_score = np.mean(parent_confidences) * 0.9  # Expect slight decay
        else:
            expected_score = 0.5  # Neutral for orphans
            
        # Get adaptive RL weights from meta-controller
        system_state = self.meta_controller.compute_system_state(
            self.epistemic_memory, 
            self.multihead_rl,
            []
        )
        adaptive_weights = self.meta_controller.adapt_weights(system_state)
        
        # Phase 6: Cultural Bias Application
        # If the parent concept belongs to a species, apply its cultural biases to the physics
        if parent_ids:
            parent_id = parent_ids[0]
            if parent_id in self.ecologist.node_to_species:
                species_id = self.ecologist.node_to_species[parent_id]
                if species_id in self.ecologist.species:
                    culture = self.ecologist.species[species_id].culture
                    for head_name, bias in culture.aesthetic_biases.items():
                        if head_name in adaptive_weights:
                            adaptive_weights[head_name] *= bias
                    
                    # Normalize weights again after cultural distortion
                    total = sum(adaptive_weights.values())
                    if total > 0:
                        adaptive_weights = {k: v/total for k, v in adaptive_weights.items()}

        # Shared multi-signal context for expanded reward heads
        reality_distance = self.epistemic_memory._compute_reality_distance(embedding_np)
        aesthetic_scores = self.vision.score_aesthetics(embedding_tensor)
        tags = self.vision.generate_tags(embedding_tensor)
        observer_boost = 0.0
        if getattr(self.meta_controller, "attention_energy", 0) > 0.05:
            attended = self.meta_controller.attended_concept
            if attended is not None and (attended in parent_ids or attended in self.active_concepts):
                observer_boost = float(self.meta_controller.attention_energy)
            elif attended is not None:
                # Soft boost if visually near the attended concept
                att_row = self.memory.get_concept(attended)
                if att_row:
                    att_emb = np.array(json.loads(att_row[1]), dtype=np.float32)
                    dist = float(np.linalg.norm(embedding_np.flatten() - att_emb.flatten()))
                    observer_boost = float(self.meta_controller.attention_energy) * max(0.0, 1.0 - dist / 2.0)

        # Curiosity uses recent system PE as prior (actual PE computed after scoring)
        pe_hist = self.meta_controller.prediction_error_history
        prior_pe = float(np.mean(pe_hist[-10:])) if pe_hist else 0.3

        # Multi-head RL evaluation with adaptive weights + interlinked signals
        total_reward, head_rewards, weights = self.multihead_rl.evaluate_concept(
            embedding_np, latent, self.memory, parent_ids=parent_ids,
            custom_weights=adaptive_weights,
            vision=self.vision,
            reality_distance=reality_distance,
            prediction_error=prior_pe,
            observer_boost=observer_boost,
            tags=tags,
            aesthetic_scores=aesthetic_scores,
        )
        
        # Actual score
        actual_score = total_reward
        
        # Prediction error (learning signal!)
        prediction_error = abs(actual_score - expected_score)
        
        # Record for meta-controller
        self.meta_controller.record_observation(
            novelty=head_rewards.get("Novelty", 0.5),
            entropy=0.7,  # Placeholder - would compute from embeddings
            prediction_error=prediction_error
        )
        
        # Combined metrics
        eval_metrics = {
            "total_reward": total_reward,
            "head_rewards": head_rewards,
            "weights": adaptive_weights,  # Show adaptive weights
            "coherence": head_rewards.get("Coherence", 0.5),
            "novelty": head_rewards.get("Novelty", 0.5),
            "familiarity": head_rewards.get("Memory", 0.5),
            "score": total_reward,
            "expected_score": expected_score,
            "prediction_error": prediction_error,
            "aesthetic": aesthetic_scores,
            "reality_distance": reality_distance,
            "reward_facets": getattr(self.multihead_rl, "last_facets", {}),
        }
        
        eval_metrics["tags"] = tags

        # Dynamic acceptance threshold from meta-controller
        acceptance_threshold = self.meta_controller.get_validation_strictness(system_state)
        accepted = total_reward > acceptance_threshold
        
        # Phase 1 & 4: Record Transformation Delta and Immediate Failures
        if len(parent_ids) == 1:
            parent_id = parent_ids[0]
            parent_latent_t = self.get_latent_for_concept(parent_id)
            if parent_latent_t is not None:
                parent_latent = parent_latent_t.cpu().numpy()
                child_latent_np = latent.cpu().numpy()
                
                # Observe the mutation to build the physics engine
                failure_type = None if accepted else FailureType.INVALID
                trans = self.physics_engine.observe_mutation(
                    parent_latent, child_latent_np, parent_id, accepted, 
                    offspring_fitness=total_reward, novelty=head_rewards.get("Novelty", 0.5), 
                    failure_type=failure_type
                )
                if trans:
                    eval_metrics["transformation_tier"] = trans.tier.value
        
        if accepted:
            # Determine generator version
            # -1 = real image (always store latent)
            # 0 = legacy stored latent (for backward compatibility)
            # 1 = current generator version (latent generated on-demand)
            if is_real_image:
                generator_version = -1  # Real images always store latents
                latent_to_store = latent.cpu().numpy()
            else:
                generator_version = 1  # Use generator for evolved concepts
                latent_to_store = None  # Don't store! Will be generated on-demand
                
                # Train generator on this new concept (incremental learning)
                # This helps the generator learn the evolved latent space
                try:
                    # Deep copy tensors and convert to numpy BEFORE scheduler submission
                    # This ensures data is fully isolated from the calling thread
                    safe_embedding = embedding_np.reshape(1, -1).copy()
                    safe_latent = latent.clone().detach().reshape(1, 4, 64, 64).cpu().numpy().copy()
                    safe_reward = np.array([total_reward])
                    
                    # Get the training scheduler (ensures all PyTorch ops on single thread)
                    scheduler = get_training_scheduler()
                    
                    # Submit training tasks to the dedicated training thread
                    # This prevents C++ race conditions in PyTorch's Adam optimizer
                    
                    # 1. Supervised: Learn the exact mapping from the accepted mutation
                    scheduler.submit(
                        self.latent_generator_trainer.train_on_batch,
                        safe_embedding,
                        safe_latent,
                        stage="supervised",
                        wait=False  # Non-blocking, queued for execution
                    )
                    
                    # 2. Self-Supervised: Ensure it understands the VAE manifold
                    # This runs less frequently to save compute
                    if self.age % 5 == 0:
                        scheduler.submit(
                            self.latent_generator_trainer.train_on_batch,
                            safe_embedding,
                            stage="roundtrip",
                            wait=False
                        )
                        
                    # 3. RL-Guided: Train generator to actively maximize fitness heads
                    # We pass the actual reward it received so it learns what makes a "good" latent
                    scheduler.submit(
                        self.latent_generator_trainer.train_on_batch,
                        safe_embedding,
                        rewards=safe_reward,
                        stage="rl",
                        wait=False
                    )
                except Exception as e:
                    # Don't fail if training fails
                    print(f"Warning: Generator training failed: {e}")
            
            # Add to database
            c_id = self.memory.add_concept(
                embedding=embedding_np,
                latent=latent_to_store,
                parent_ids=parent_ids,
                generation=generation,
                coherence=head_rewards.get("Coherence", 0.5),
                novelty=head_rewards.get("Novelty", 0.5),
                tags=tags,
                generator_version=generator_version
            )
            
            # Add to epistemic memory (confidence-based)
            if is_real_image:
                conf_obj = self.epistemic_memory.add_real_image(c_id, embedding_np)
            else:
                conf_obj = self.epistemic_memory.add_derived_concept(
                    c_id, embedding_np, parent_confidences
                )
                
            # Validation loop: decode → re-encode → check reality distance
            conf_obj.confidence = self.epistemic_memory.validate_concept(
                c_id, latent, expected_score, actual_score
            )
            
            eval_metrics["confidence"] = conf_obj.confidence
            eval_metrics["reality_distance"] = conf_obj.reality_distance
            eval_metrics["tier"] = conf_obj.tier.value
            
            # PHASE 1: Add node to concept graph
            graph_tier = NodeTier.REALITY if is_real_image else NodeTier.DREAM
            graph_node = ConceptNode(
                id=c_id,
                embedding=embedding_np,
                tier=graph_tier,
                confidence=conf_obj.confidence,
                age=generation,
                fitness=total_reward,
                latent=latent.cpu().numpy()
            )
            
            # Phase 6: Attach ecology profile
            graph_node.ecology = ecology
            
            # Add parent edges
            for parent_id in parent_ids:
                if parent_id in self.concept_graph.nodes:
                    graph_node.parents.append(parent_id)
                    
            self.concept_graph.add_node(graph_node)
            
            # Add evolved_from edges
            for parent_id in parent_ids:
                self.concept_graph.add_edge(parent_id, c_id, EdgeType.EVOLVED_FROM)
                
            # Compute similarity edges to build the neighborhood
            self.concept_graph.compute_similarity_edges(c_id, k=5)
                
            # If this was a recombination, record combines_with
            if len(parent_ids) == 2:
                self.concept_graph.add_combination_edge(
                    parent_ids[0], parent_ids[1], c_id, total_reward
                )
                
            # Record outcome for GNN training
            if parent_ids:
                self.gnn.record_offspring_quality(parent_ids[0], total_reward)
                if len(parent_ids) == 2:
                    self.gnn.record_combination_success(
                        parent_ids[0], parent_ids[1], total_reward
                    )
            
            self.active_concepts = [c_id]
            self.current_latent = latent
            self.current_image = self.vision.decode_latent_to_image(latent)
            
            # Log images sparsely
            if self.image_logger.should_save_milestone(self.age):
                self.image_logger.save_milestone(
                    self.current_image, 
                    self.age, 
                    c_id,
                    metadata={
                        'fitness': total_reward,
                        'confidence': conf_obj.confidence,
                        'tier': conf_obj.tier.value,
                        'mode': 'unknown'  # Will be set by caller
                    }
                )
            
            # Log high fitness concepts
            self.image_logger.save_high_fitness(
                self.current_image, self.age, c_id, total_reward
            )
            
            self.age += 1

            # --- Staggered maintenance (never pile GNN + extinction + FAISS + persist) ---
            #   % 100      → GNN train only
            #   % 100==25  → FAISS ghost flush
            #   % 100==50  → promote / quarantine / extinction (capped)
            #   % 100==75  → full memory persist (+ leftover FAISS flush)
            tick = self.age % 100

            if tick == 0 and self.age > 0:
                try:
                    print(f"\n[GNN] Training at generation {self.age}...")
                    loss = self.gnn.train_step(max_nodes=256)
                    if loss > 0:
                        print(f"[GNN Training] Loss: {loss:.4f}")
                    # train_step already persists weights when loss > 0
                except Exception as e:
                    print(f"[GNN] Periodic train failed (continuing): {e}")

            elif tick == 25:
                if self.faiss_ghosts > 0:
                    self._maybe_rebuild_faiss(force=True)

            elif tick == 50:
                promoted = self.epistemic_memory.promote_frontier_concepts()
                quarantined = self.epistemic_memory.quarantine_dangerous_concepts()

                for promoted_id in promoted:
                    if self.current_image and self.active_concepts and promoted_id in self.active_concepts:
                        self.image_logger.save_promotion(
                            self.current_image, self.age, promoted_id,
                            from_tier="frontier", to_tier="working"
                        )

                # Cap mass extinction so we don't kill+rebuild hundreds in one frame
                extinct_count = self._run_extinction_sweep(max_prune=50)

                if promoted or quarantined or extinct_count > 0:
                    print(
                        f"\n[Ecosystem Update] Promoted: {len(promoted)}, "
                        f"Quarantined: {len(quarantined)}, Extinct: {extinct_count}"
                    )
                    stats = self.image_logger.get_stats()
                    print(
                        f"[ImageLogger] Saved - Milestones: {stats['milestones']}, "
                        f"Promotions: {stats['promotions']}, "
                        f"High Fitness: {stats['high_fitness']}, "
                        f"Combinations: {stats['combinations']}"
                    )
                    if extinct_count > 0:
                        self._ensure_valid_active_concepts()
                        # Defer FAISS rebuild to tick 25/75 — don't rebuild here

            elif tick == 75:
                try:
                    if self.faiss_ghosts > 0:
                        self._maybe_rebuild_faiss(force=True)
                    self._persist_memory_layers(reason="periodic_75")
                except Exception as e:
                    print(f"[Memory] Periodic save failed (continuing): {e}")

            return True, eval_metrics
        return False, eval_metrics

    def _select_concepts_for_evolution(self, count: int = 2):
        """
        PHASE 2: Graph + GNN guided selection with exploration safeguard.
        
        Strategy:
        - Use GNN recommendations (learned patterns)
        - Add exploration bonus (prevent getting stuck)
        - Meta-controller adjusts GNN influence
        - Keep random exploration budget
        
        The GNN is a guide, not a ruler.
        """
        # Fallback if no concepts
        if len(self.concept_graph.nodes) < count:
            return [self.active_concepts[0]] if self.active_concepts else []
            
        # Meta-controller decides GNN influence ratio
        system_state = self.meta_controller.compute_system_state(
            self.epistemic_memory, self.multihead_rl, []
        )
        exploration_mode = self.meta_controller.get_exploration_mode(system_state)
        
        # Adjust GNN influence based on mode (reduced values to increase exploration)
        if exploration_mode == "EXPLOIT":
            self.gnn_influence = 0.50  # Trust GNN more when exploiting (reduced from 0.70)
        elif exploration_mode == "EXPLORE":
            self.gnn_influence = 0.20  # Trust GNN less when exploring (reduced from 0.30)
        else:  # BALANCED
            self.gnn_influence = 0.30  # Reduced from 0.50
            
        # Decide: GNN-guided or random exploration?
        if np.random.random() < self.gnn_influence:
            # GNN-GUIDED SELECTION
            recommendations = self.gnn.recommend_parents_for_mutation(
                k=min(20, len(self.concept_graph.nodes)),
                current_time=self.age
            )
            
            if not recommendations:
                # Fallback to random
                all_ids = list(self.concept_graph.nodes.keys())
                return list(np.random.choice(all_ids, min(count, len(all_ids)), replace=False))
                
            # Extract recommendations with scores — only living graph nodes
            # Format: (node_id, score, uncertainty, exploration_bonus)
            candidates = [
                (nid, score) for nid, score, _, _ in recommendations
                if nid in self.concept_graph.nodes and self.memory.concept_exists(nid)
            ]
            if not candidates:
                all_ids = [i for i in self.concept_graph.nodes.keys() if self.memory.concept_exists(i)]
                if not all_ids:
                    return list(self.active_concepts)[:count]
                return list(np.random.choice(all_ids, min(count, len(all_ids)), replace=False))

            node_ids = [nid for nid, _ in candidates]
            scores = np.array([score for _, score in candidates])
            scores = np.maximum(scores, 0.0)
            if scores.sum() == 0:
                probs = np.ones(len(scores)) / len(scores)
            else:
                probs = scores / scores.sum()

            selected_ids = list(np.random.choice(
                node_ids,
                min(count, len(node_ids)),
                p=probs,
                replace=False
            ))
            
            # Update visit tracking
            for nid in selected_ids:
                if nid in self.concept_graph.nodes:
                    node = self.concept_graph.nodes[nid]
                    node.selection_count += 1
                    node.last_visited = self.age
                    node.visit_frequency = node.selection_count / (self.age + 1)
                    
        else:
            # RANDOM EXPLORATION ("dark matter" budget)
            # Explore regions the GNN hasn't recommended
            all_ids = [i for i in self.concept_graph.nodes.keys() if self.memory.concept_exists(i)]
            if not all_ids:
                return list(self.active_concepts)[:count]
            
            # Weight by inverse visit frequency (prefer unexplored)
            weights = []
            for nid in all_ids:
                node = self.concept_graph.nodes[nid]
                # Inverse visit frequency + exploration bonus
                inverse_freq = max(0.0, 1.0 - node.visit_frequency)  # Ensure non-negative
                exploration = max(0.0, node.exploration_bonus(self.age))  # Ensure non-negative
                weight = inverse_freq + exploration + 0.01  # Add small constant to avoid zeros
                weights.append(weight)
                
            weights = np.array(weights)
            # Ensure all weights are positive
            weights = np.maximum(weights, 0.01)
            
            if weights.sum() == 0:
                probs = np.ones(len(weights)) / len(weights)
            else:
                probs = weights / weights.sum()
                
            selected_ids = list(np.random.choice(
                all_ids,
                min(count, len(all_ids)),
                p=probs,
                replace=False
            ))
            
        return selected_ids
    
    def _run_competition_event(self, required_energy: float) -> bool:
        """
        Phase 6: Scarcity.
        When the universe is out of energy, a new concept must outcompete an existing one.
        Finds the weakest concept based on the survival_pressure equation and kills it to free energy.
        """
        if len(self.concept_graph.nodes) < 100:
            # Don't cannibalize if the universe is tiny
            return False
            
        weakest_node_id = None
        lowest_pressure = float('inf')
        
        for node_id, node in self.concept_graph.nodes.items():
            if node.tier == NodeTier.REALITY:
                continue # Reality anchors are immune
                
            # Calculate Survival Pressure
            # survival_pressure = fitness + ecology + culture + history - cost - redundancy - instability
            
            biological_fitness = node.fitness
            ecological_value = node.ecology.ecological_value
            
            # Cultural attention (is it part of a species that is currently active?)
            cultural_attention = 0.0
            if node_id in self.ecologist.node_to_species:
                s_id = self.ecologist.node_to_species[node_id]
                if s_id in self.ecologist.species:
                    s = self.ecologist.species[s_id]
                    # If this species was recently active, it has cultural attention
                    if s.members.intersection(set(self.active_concepts)):
                        cultural_attention = 0.5
            
            # Historical importance (graph centrality proxy: offspring count)
            historical_importance = min(1.0, node.offspring_count / 10.0)
            
            # Costs
            cost = node.ecology.energy_cost + node.ecology.memory_cost
            redundancy = node.ecology.redundancy
            instability = node.mean_prediction_error()
            
            survival_pressure = (biological_fitness + ecological_value + cultural_attention + historical_importance) - (cost + redundancy + instability)
            
            if survival_pressure < lowest_pressure:
                lowest_pressure = survival_pressure
                weakest_node_id = node_id
                
        if weakest_node_id is not None:
            weak_node = self.concept_graph.nodes[weakest_node_id]
            freed_energy = weak_node.ecology.energy_cost
            
            # Kill the weak concept
            print(f"  -> Concept {weakest_node_id} outcompeted (Pressure: {lowest_pressure:.2f}). Freed {freed_energy:.1f} energy.")
            self._kill_concept(weakest_node_id, "Outcompeted for resources", EdgeType.FAILED_STERILE)
            
            self.global_energy_pool += freed_energy
            
            # Did we free enough?
            if self.global_energy_pool >= required_energy:
                return True
                
        return False

    def _maybe_rebuild_faiss(self, force: bool = False):
        """
        Rebuild FAISS mid-session when enough extinct vectors have piled up.
        Lookups stay correct without this (id_map + living filter), but the
        index still wastes slots until rebuild — do it before restart.
        """
        if not force and self.faiss_ghosts < self.faiss_rebuild_threshold:
            return False
        if self.faiss_ghosts <= 0 and not force:
            return False
        print(f"[FAISS] Mid-run rebuild ({self.faiss_ghosts} ghosts)...")
        self.memory.rebuild_faiss_from_db()
        self.faiss_ghosts = 0
        return True

    def _kill_concept(self, node_id: int, reason: str, failure_type: EdgeType = None):
        """Helper to properly remove a concept from all systems."""
        if node_id not in self.concept_graph.nodes:
            return
            
        node = self.concept_graph.nodes[node_id]
        
        # Record hazard edge
        if failure_type:
            for parent_id in node.parents:
                self.concept_graph.add_edge(parent_id, node_id, failure_type)
                self.gnn.record_hazard(parent_id, failure_type.value)
                
        # Remove from database (also drops id_map slot — lookups skip this ID immediately)
        if self.memory.prune_concept(node_id, reason):
            self.faiss_ghosts += 1
        
        # Remove from Concept Graph
        self.concept_graph.quarantine_node(node_id)
        
        # Remove from Epistemic Memory
        if node_id in self.epistemic_memory.working_concepts:
            del self.epistemic_memory.working_concepts[node_id]
        if node_id in self.epistemic_memory.frontier_concepts:
            del self.epistemic_memory.frontier_concepts[node_id]
            
        # Update Active Concepts
        if node_id in self.active_concepts:
            self.active_concepts.remove(node_id)
            if not self.active_concepts and self.concept_graph.reality_nodes:
                self.active_concepts = [list(self.concept_graph.reality_nodes)[0]]

        # Don't rebuild on every scarcity kill — batch when ghosts pile up
        self._maybe_rebuild_faiss(force=False)

    def _run_extinction_sweep(self, max_prune: int = 50) -> int:
        """
        Evaluate all non-reality concepts for extinction.
        Phase 4: Classifies failures (UNSTABLE, STERILE, CATASTROPHIC) and adds hazard edges.
        max_prune: hard cap per sweep so maintenance doesn't freeze/kill the process.
        """
        extinct_count = 0
        concepts_to_prune = []
        
        # Don't prune if we have very few concepts
        if len(self.concept_graph.nodes) < 50:
            return 0
        
        # Create snapshot to avoid dictionary modification during iteration
        nodes_snapshot = list(self.concept_graph.nodes.items())
            
        for node_id, node in nodes_snapshot:
            # Never prune Reality anchors
            if node.tier == NodeTier.REALITY:
                continue
                
            # Don't prune brand new concepts (give them a chance)
            if self.age - node.age < 50:
                continue
                
            reason = None
            failure_type = None
            
            # Rule 1: UNSTABLE (High prediction error / Collapsed)
            if node.mean_prediction_error() > 0.7 and node.validation_count > 10:
                reason = "Unlearnable (High Pred Error)"
                failure_type = EdgeType.FAILED_UNSTABLE
                
            # Rule 2: STERILE (Low fitness / dead end / no descendants)
            elif node.fitness < 0.15 and node.offspring_count == 0 and (self.age - node.age) > 100:
                reason = "Sterile (Dead End)"
                failure_type = EdgeType.FAILED_STERILE
                
            # Rule 3: CATASTROPHIC (Destroyed lineage - produced many bad children)
            elif node.offspring_count > 5 and node.offspring_success_rate() < 0.1 and (self.age - node.age) > 100:
                reason = "Catastrophic (Toxic Lineage)"
                failure_type = EdgeType.FAILED_CATASTROPHIC
                
            # Rule 4: Forgotten (Old and never visited)
            elif node.visit_frequency < 0.005 and (self.age - node.age) > 150:
                reason = "Forgotten (Never Visited)"
                # Not a hazard, just garbage collection
                
            if reason:
                concepts_to_prune.append((node_id, reason, failure_type))

        # Oldest / lowest fitness first, then cap
        concepts_to_prune.sort(
            key=lambda t: (
                self.concept_graph.nodes[t[0]].fitness if t[0] in self.concept_graph.nodes else 0.0,
                -(self.age - (self.concept_graph.nodes[t[0]].age if t[0] in self.concept_graph.nodes else 0)),
            )
        )
        concepts_to_prune = concepts_to_prune[:max_prune]

        # Execute pruning
        for node_id, reason, failure_type in concepts_to_prune:
            # Phase 4: Record the hazard edge from parents before deleting
            if failure_type and node_id in self.concept_graph.nodes:
                node = self.concept_graph.nodes[node_id]
                for parent_id in node.parents:
                    self.concept_graph.add_edge(parent_id, node_id, failure_type)
                    
            # 1. Remove from database
            if self.memory.prune_concept(node_id, reason):
                self.faiss_ghosts += 1
                # Phase 4: Train GNN to avoid this hazard
                if failure_type:
                    node = self.concept_graph.nodes.get(node_id)
                    parents = node.parents if node else []
                    for parent_id in parents:
                        self.gnn.record_hazard(parent_id, failure_type.value)
                        
                # 2. Remove from Concept Graph
                if node_id in self.concept_graph.nodes:
                    self.concept_graph.quarantine_node(node_id)
                # 3. Remove from Epistemic Memory
                if node_id in self.epistemic_memory.working_concepts:
                    del self.epistemic_memory.working_concepts[node_id]
                if node_id in self.epistemic_memory.frontier_concepts:
                    del self.epistemic_memory.frontier_concepts[node_id]
                    
                extinct_count += 1
                
                # 4. Update Active Concepts if necessary
                if node_id in self.active_concepts:
                    self.active_concepts.remove(node_id)
                    if not self.active_concepts:
                        # Fallback to a reality node
                        if self.concept_graph.reality_nodes:
                            self.active_concepts = [list(self.concept_graph.reality_nodes)[0]]
                            
        return extinct_count

    def _select_pairs_for_recombination(self):
        """
        Uses GNN to recommend pairs that are likely to combine successfully.
        """
        if len(self.concept_graph.nodes) < 2:
            return []
            
        system_state = self.meta_controller.compute_system_state(
            self.epistemic_memory, self.multihead_rl, []
        )
        exploration_mode = self.meta_controller.get_exploration_mode(system_state)
        
        if exploration_mode == "EXPLOIT":
            self.gnn_influence = 0.50  # Reduced from 0.70
        elif exploration_mode == "EXPLORE":
            self.gnn_influence = 0.20  # Reduced from 0.30
        else:
            self.gnn_influence = 0.30  # Reduced from 0.50
            
        if np.random.random() < self.gnn_influence:
            recommendations = self.gnn.recommend_parent_pairs_for_recombination(
                k=10, current_time=self.age
            )
            if recommendations:
                # Random weighted choice from top pairs
                candidates = [(a, b, score) for a, b, score, _ in recommendations]
                scores = np.array([s for _, _, s in candidates])
                scores = np.maximum(scores, 0.0)
                if scores.sum() > 0:
                    probs = scores / scores.sum()
                    chosen_idx = np.random.choice(len(candidates), p=probs)
                    a, b, _ = candidates[chosen_idx]
                    return [a, b]
                    
        # Fallback to random pairs
        all_ids = list(self.concept_graph.nodes.keys())
        if len(all_ids) >= 2:
            return list(np.random.choice(all_ids, 2, replace=False))
        return []

    def inject_attention(self, concept_id: int):
        """
        Phase 5: Observer Attention.
        The user clicked a concept. We don't just clone it; we inject attention energy
        into the meta-controller to force curiosity and exploration around this node.
        """
        with self.lock:
            if not self.memory.concept_exists(concept_id):
                print(f"[Observer] Concept {concept_id} no longer exists")
                return False
            if concept_id in self.concept_graph.nodes:
                print(f"\n[Observer] Attention injected into Concept {concept_id}")
                self.meta_controller.focus_attention(concept_id)
                
                # Set as active concept so evolution starts from here
                self.active_concepts = [concept_id]
                
                lat = self.get_latent_for_concept(concept_id)
                if lat is None:
                    return False
                self.current_latent = lat
                self.current_image = self.vision.decode_latent_to_image(self.current_latent)
                return True
        return False

    def _active_verb_discovery(self):
        """
        Actively calculates the semantic vector between two highly successful concepts.
        If Concept A is great, and Concept B is great, the vector A->B is a recipe 
        for 'How to make A more like B'.
        """
        print("\n[Physics] Actively researching new verbs via Latent Arithmetic...")
        # Get the top 30 most fit, living nodes
        top_nodes = sorted(
            [n for n in self.concept_graph.nodes.values() if n.tier != NodeTier.REALITY and self.memory.concept_exists(n.id)], 
            key=lambda n: n.fitness, reverse=True
        )[:30]
        
        if len(top_nodes) < 2:
            return
            
        import random
        extracted = 0
        for _ in range(3): # Try to extract up to 3 new hypothesis verbs
            a = random.choice(top_nodes)
            b = random.choice(top_nodes)
            if a.id == b.id: continue
            
            lat_a = self.get_latent_for_concept(a.id)
            lat_b = self.get_latent_for_concept(b.id)
            
            if lat_a is not None and lat_b is not None:
                verb = self.physics_engine.extract_semantic_verb(
                    lat_a.cpu().numpy(), lat_b.cpu().numpy()
                )
                if verb:
                    print(f"  -> Extracted Hypothesis Verb {verb.id} (Vector: Concept {a.id} -> {b.id})")
                    extracted += 1
                    
        if extracted == 0:
            print("  -> No viable verbs extracted.")

    def step(self):
        """
        Advances the universe by one step.
        
        Uses multi-head RL with curiosity-driven concept selection.
        Implements both mutation (70%) and recombination (30%).
        """
        with self.lock:
            if self.current_latent is None:
                return False, {}

            # Never evolve from extinct / ghost IDs
            if not self._ensure_valid_active_concepts():
                print("[Evolve] No living concepts available.")
                return False, {}
                
            import random
        
            # Add energy replenishment and deduct maintenance
            # Scarcity mechanics:
            # - Birth costs ~4 energy
            # - Replenish gives +25 energy
            # - Maintenance costs 0.01 per node
            # This causes the universe to naturally cap at around 2100 concepts!
            maintenance_cost = len(self.concept_graph.nodes) * 0.01
            net_energy = self.energy_replenishment_rate - maintenance_cost
            self.global_energy_pool = max(0.0, min(self.max_energy_pool, self.global_energy_pool + net_energy))
            
            # Evolve the verbs themselves periodically
            if self.age % 200 == 0 and self.age > 0:
                self.physics_engine.evolve_verbs()
                
            # Phase 6: Active Latent Arithmetic Discovery
            if self.age % 150 == 0 and self.age > 0:
                self._active_verb_discovery()
                
            # Phase 6: Deep Time - Decay unused laws
            if self.age % 500 == 0 and self.age > 0:
                self.physics_engine.decay_laws(self.age)
                
            # Phase 6: Ecology - Discover species and check for speciation
            if self.age % 1000 == 0 and self.age > 0:
                self.ecologist.discover_species(self.concept_graph, self.physics_engine, self.age)
                
            # Phase 6: History - Compress extinct species into Legends
            if self.age % 2000 == 0 and self.age > 0:
                self.ecologist.run_historian_sweep(self.concept_graph, self.physics_engine)
            
            # Decide: mutation or recombination?
            new_latent = None
            parent_ids = list(self.active_concepts)
            mode = "mutation"

            if random.random() < 0.5 and self.memory.index.ntotal > 10:
                # RECOMBINATION: Fuse distant concepts
                selected_ids = self._select_pairs_for_recombination()
                latent1 = latent2 = None
                valid_parents = []
                for c_id in selected_ids:
                    cid = int(c_id)
                    if not self.memory.concept_exists(cid):
                        continue
                    lat = self.get_latent_for_concept(cid)
                    if lat is None:
                        continue
                    if latent1 is None:
                        latent1 = lat
                        valid_parents.append(cid)
                    else:
                        latent2 = lat
                        valid_parents.append(cid)
                        break
                if latent1 is not None and latent2 is not None and len(valid_parents) >= 2:
                    alpha = random.uniform(0.3, 0.7)
                    new_latent = self.dreamer.recombine_latents(latent1, latent2, alpha=alpha)
                    parent_ids = valid_parents[:2]
                    mode = "recombination"

            if new_latent is None:
                # MUTATION: Explore locally (also fallback when recombination fails)
                base_latent = self.current_latent
                parent_ids = list(self.active_concepts)
                if random.random() < 0.2 and self.memory.index.ntotal > 5:
                    selected_ids = self._select_concepts_for_evolution(count=1)
                    if selected_ids:
                        cid = int(selected_ids[0])
                        if self.memory.concept_exists(cid):
                            candidate = self.get_latent_for_concept(cid)
                            if candidate is not None:
                                base_latent = candidate
                                parent_ids = [cid]

                if base_latent is None:
                    print("[Evolve] No base latent available.")
                    return False, {}

                available_energy = 0.5
                if parent_ids:
                    parent_conf = self.epistemic_memory._get_concept(parent_ids[0])
                    if parent_conf:
                        available_energy = parent_conf.get_mutation_energy(self.age)

                verb_candidates = self.physics_engine.get_verb_candidates(
                    available_energy=available_energy
                )

                if (
                    random.random() < 0.05
                    and self.ecologist.legends
                    and getattr(self.meta_controller, "attention_energy", 0) > 0.5
                ):
                    legends = list(self.ecologist.legends.values())
                    weights = np.array([l.mythic_weight for l in legends], dtype=np.float64)
                    if weights.sum() > 0:
                        probs = weights / weights.sum()
                        chosen_legend = legends[int(np.random.choice(len(legends), p=probs))]
                        for verb_id in chosen_legend.dominant_verbs:
                            if verb_id in self.physics_engine.transformations:
                                verb = self.physics_engine.transformations[verb_id]
                                if verb.tier == FailureType.FOSSILIZED_LAW or verb.confidence < 0.2:
                                    print(
                                        f"\n[Archaeology] Resurrecting fossilized verb {verb.id} "
                                        f"inspired by Legend {chosen_legend.id}!"
                                    )
                                    verb.confidence = 0.5
                                    verb_candidates.insert(0, verb)
                                    break

                if verb_candidates and random.random() < 0.90:
                    if random.random() < 0.78:
                        top_k = max(1, int(len(verb_candidates) * 0.2))
                        chosen_verb = random.choice(verb_candidates[:top_k])
                        delta_tensor = torch.tensor(chosen_verb.delta_latent, dtype=torch.float32).to(self.device)
                        if delta_tensor.dim() == 3 and base_latent.dim() == 4:
                            delta_tensor = delta_tensor.unsqueeze(0)
                            
                        # Guard against bad tensors
                        if not torch.isfinite(delta_tensor).all() or not torch.isfinite(base_latent).all():
                            return False, {}
                            
                        new_latent = base_latent + delta_tensor
                        mode = f"verb_mutation (id:{chosen_verb.id})"
                    else:
                        chosen_verb = random.choice(verb_candidates)
                        temp_mutated_verb = chosen_verb.mutate(-1)
                        delta_tensor = torch.tensor(temp_mutated_verb.delta_latent, dtype=torch.float32).to(self.device)
                        if delta_tensor.dim() == 3 and base_latent.dim() == 4:
                            delta_tensor = delta_tensor.unsqueeze(0)
                            
                        # Guard against bad tensors
                        if not torch.isfinite(delta_tensor).all() or not torch.isfinite(base_latent).all():
                            return False, {}
                            
                        new_latent = base_latent + delta_tensor
                        mode = f"mutated_verb_mutation (parent_id:{chosen_verb.id})"
                else:
                    new_latent = self.dreamer.mutate_latent(base_latent, mutation_rate=0.30)
                    if not torch.isfinite(new_latent).all():
                        return False, {}
                    mode = "quantum_fluctuation"

            # Drop any parent that vanished between selection and process
            parent_ids = [int(p) for p in parent_ids if self.memory.concept_exists(int(p))]
            if not parent_ids and self.active_concepts:
                parent_ids = [
                    int(p) for p in self.active_concepts if self.memory.concept_exists(int(p))
                ]

            success, metrics = self._process_new_concept(
                new_latent,
                parent_ids=parent_ids,
                generation=self.age,
                is_real_image=False,
            )
            
            # Phase 7: Save generator checkpoints periodically
            if self.age % 500 == 0 and self.age > 0:
                # Use scheduler to ensure checkpoint saving doesn't conflict with training
                scheduler = get_training_scheduler()
                scheduler.submit(
                    self.latent_generator_trainer.save_checkpoint,
                    wait=True  # Block to ensure checkpoint is saved before continuing
                )
                
            if metrics is None:
                metrics = {}
            metrics["mode"] = mode

            if success and mode == "recombination" and len(parent_ids) == 2:
                try:
                    self.image_logger.save_combination(
                        self.current_image,
                        self.age,
                        self.active_concepts[0] if self.active_concepts else 0,
                        parent_ids[0],
                        parent_ids[1],
                        metrics.get("total_reward", 0),
                    )
                except Exception as e:
                    print(f"[ImageLogger] combination save skipped: {e}")

            return success, metrics

    def ingest_image(self, image_path: str):
        """
        Ingests a real image into the universe memory.
        Real images go directly to LANDMARK tier with full confidence.
        """
        from PIL import Image
        with self.lock:
            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize((512, 512))
                
                # Encode to latent
                latent = self.vision.encode_image_to_latent(img)
                
                # Extract concept directly from image for higher fidelity
                embedding_tensor = self.vision.extract_concept(img)
                embedding_np = embedding_tensor.cpu().numpy().flatten()  # Ensure 1D array
                
                # Check if already ingested (distance near 0)
                nearest = self.memory.get_nearest_concepts(embedding_np, k=1)
                if nearest and len(nearest) > 0 and nearest[0][1] < 1e-4:
                    # Already in memory, just return the existing concept ID
                    return True, nearest[0][0]
                
                # Generate tags for real image
                tags = self.vision.generate_tags(embedding_tensor)

                # Add to memory directly as a "ground truth" anchor concept
                # Ensure latent is correct shape [4, 64, 64] not [1, 4, 64, 64]
                latent_np = latent.cpu().numpy()
                if latent_np.ndim == 4 and latent_np.shape[0] == 1:
                    latent_np = latent_np[0]  # Remove batch dimension
                
                c_id = self.memory.add_concept(
                    embedding=embedding_np,
                    latent=latent_np,
                    parent_ids=[],
                    generation=self.age,
                    coherence=1.0,  # Real images have perfect coherence
                    novelty=1.0,     # Treat as highly novel to encourage exploration
                    tags=tags,
                    generator_version=-1  # Real images always store latents
                )
                
                # Add to epistemic memory as LANDMARK (immutable reality anchor)
                self.epistemic_memory.add_real_image(c_id, embedding_np)
                
                # Add to concept graph as REALITY node (root)
                reality_node = ConceptNode(
                    id=c_id,
                    embedding=embedding_np,
                    tier=NodeTier.REALITY,
                    confidence=1.0,
                    age=self.age,
                    fitness=1.0,
                    latent=latent.cpu().numpy()
                )
                self.concept_graph.add_node(reality_node)
                
                # Optionally set as active concept to shift the dream
                self.active_concepts = [c_id]
                self.current_latent = latent
                self.current_image = img
                # DON'T increment age during ingestion - only during evolution!
                
                return True, c_id
            except Exception as e:
                print(f"Failed to ingest {image_path}: {e}")
                return False, None

    def background_ingest_step(self):
        """Ingests one image from the dataset, designed to be called in the background loop."""
        import os
        with self.lock:
            dataset_dir = "dataset"
            if not os.path.exists(dataset_dir):
                return False
                
            valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
            all_files = [f for f in os.listdir(dataset_dir) 
                         if os.path.splitext(f)[1].lower() in valid_exts]
            
            import random
            random.shuffle(all_files)
                         
            # Find one un-ingested file
            for filename in all_files:
                if not self.memory.is_file_ingested(filename):
                    filepath = os.path.join(dataset_dir, filename)
                    success, c_id = self.ingest_image(filepath)
                    if success and c_id is not None:
                        self.memory.mark_file_ingested(filename, c_id)
                    return True # We did some work
                    
            return False # All ingested
    def scan_dataset(self, dataset_dir="dataset", limit: int = None, randomize: bool = False):
        """Scans the dataset directory and ingests images, optionally limiting and randomizing the count."""
        import os
        import random
        results = []
        with self.lock:
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
                
            valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
            
            # Count total files first
            all_files = [f for f in os.listdir(dataset_dir) 
                         if os.path.splitext(f)[1].lower() in valid_exts]
            
            if randomize:
                random.shuffle(all_files)
                
            if limit:
                all_files = all_files[:limit]
                
            total_files = len(all_files)
            print(f"Found {total_files} images to ingest...")
            
            ingested_count = 0
            for i, filename in enumerate(all_files, 1):
                if self.memory.is_file_ingested(filename):
                    continue
                    
                filepath = os.path.join(dataset_dir, filename)
                success, c_id = self.ingest_image(filepath)
                if success and c_id is not None:
                    self.memory.mark_file_ingested(filename, c_id)
                    ingested_count += 1
                results.append((filename, success))
                
                # Print progress every 50 images
                if i % 50 == 0 or i == total_files:
                    print(f"Progress: {i}/{total_files} processed, {ingested_count} ingested")
                    
            return results
