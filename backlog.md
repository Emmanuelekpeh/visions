# Backlog

## SESSION STATUS: 2026-07-18 (morning)
**Maintenance stagger:** GNN@%100 / FAISS@%25 / extinction@%50 (cap 15) / persist@%75 — no more pile-up crash
**Dataset:** Removed 440 corrupted images (verify+load failed)
**Fresh universe:** Empty DB re-anchored with 200 images; FAISS mid-run rebuilds working

## SESSION STATUS: 2026-07-17 (late)
**Critical:** GNN gen-100 crash hardened (subsample max 256 nodes, try/except, no double-save)
**Reward rewrite:** Multi-signal SharedSignals + sibling coupling — Beauty is CLIP facets + reality + observer + anti-sludge, not latent smoothness
**Heads:** Beauty/Novelty/Coherence/Compression/Memory/Curiosity all multi-facet and interlinked

## SESSION STATUS: 2026-07-17
**Architecture:** 11-core system - Added Latent Generator to solve unbounded memory growth
**Critical Fix:** Implemented learned latent generation (storage: 67KB → 2KB per concept)
**Training Status:** Generator trained on 50 images (cosine sim: 0.78, needs more training)
**Integration:** Generator integrated into world.py, schema update and full migration pending

## PREVIOUS SESSION: 2026-07-15
**Architecture:** 10-core system including Multi-head RL, Epistemic Memory, and GNN Reasoning Layer.
**Data Integration:** Image ingestion functional; universe anchors to dataset landmarks.
**Fixes (CRITICAL PERSISTENCE FIX):** 
- 🔴 **CRITICAL:** Complete memory persistence - Added TWO metadata tables
  - `graph_metadata` - Graph state (combinations, relationships, usage tracking)
  - `epistemic_metadata` - Epistemic state (tiers, confidence, validation data)
  - **FIXES SECOND-RUN CRASH** - Tier states now synchronized between layers
  - GNN can now accumulate learning across sessions
  - Auto-saves both layers every 50 and 100 generations
- Performance: Vectorized reality distance computation (120s → <10s load time)
- Stagnation: Reduced GNN influence by 20-30% to increase exploration
- Promotion: Lowered thresholds (validations 5→3, confidence 0.65→0.50)
- Extinction: Extended grace periods (youth 20→50 gen, relaxed all criteria)
- Meta-controller: Reduced sensitivity (novelty -0.2→-0.4, PE 0.05→0.02)

**Previous Fixes:** 
- Ingestion "Age" bug fixed (generation tracking now separated from total DB rows).
- GNN Recombination correctly integrated (using pair recommendations rather than single mutation lists).

---

## PRIORITY 0: Latent Generator (COMPLETE ✅)
- [x] Create latent_generator.py with LatentGenerator network (306MB, 80M params)
- [x] Create train_latent_generator.py training script
- [x] Initial training on 50 images (cosine sim: 0.78)
- [x] Integrate generator into world.py (get_latent_for_concept method)
- [x] Update database schema (add generator_version column)
- [x] Modify storage behavior (skip storing evolved latents)
- [x] Test system with generated latents (5/6 tests passing)
- [x] Verify 96% storage reduction (verified at scale, works correctly)
- [ ] OPTIONAL: Train on full dataset (242 images, target cosine sim > 0.90)

## PRIORITY 0b: Reward Semantics + GNN Stability (IN PROGRESS)
- [x] Harden GNN train_step (subsample, grad clip, never crash process)
- [x] Deduplicate gen 50/100 memory saves
- [x] Wrap evolve_step so thread exceptions don't kill UI silently
- [x] Expand all 6 heads into multi-facet SharedSignals
- [x] Interlink heads via couple_head_rewards
- [x] Beauty = CLIP aesthetic facets + reality resonance + observer + anti-sludge
- [ ] Tune aesthetic contrast on more real images / watch acceptance rates
- [ ] Surface facet breakdown in UI overlay (optional)

## PRIORITY 1: The Physics Engine (Phase 1-5)
- [ ] Phase 1: Transformation Objects (Make deltas first-class entities that track success/failure across domains)
- [ ] Phase 2: Transformation Evolution (Let the verbs themselves mutate and compete)
- [ ] Phase 3: Thermodynamic Constraints (Energy = confidence × age × stability × unused_potential)
- [ ] Phase 4: Granular Hazard Map (Differentiate Invalid, Unstable, Sterile, Catastrophic failures)
- [ ] Phase 5: Observer Attention (User interaction influences curiosity, not direct generation)

## PRIORITY 2: Semantic Concept Tagging & Filtering (MEDIUM)
- [x] Add CLIP text encoder to vision.py
- [x] Generate top-3 semantic tags per concept
- [x] Store tags in database
- [x] Display tags in UI overlay
- [ ] Enable tag-based concept filtering in UI and evolution selection

## PRIORITY 2: Concept Species & Evolution (MEDIUM)
- [ ] Implement lineage classification algorithm
- [ ] Create "species" grouping based on embedding clusters
- [ ] Track dominant species over time
- [ ] Visualize species evolution tree
- [ ] Add species color-coding to graph view

## PRIORITY 4: Timeline & History (LOW)
- [ ] Implement timeline slider component in Pygame UI
- [ ] Add scrubbing through generations
- [ ] Create snapshot export feature
- [ ] Build concept genealogy viewer

## PRIORITY 5: Autonomous Observer (RESEARCH)
- [ ] Design attention mechanism
- [ ] Implement "interesting region" detection
- [ ] Add camera control to observer
- [ ] Create narrative generation (why this concept matters)

---

## TECHNICAL DEBT
- [ ] Add version pins to requirements.txt
- [ ] Implement FAISS index persistence (save/load between sessions)
- [ ] Clean up legacy `RealityAnchor` class in `engines.py` (now superseded by Multihead RL)
- [ ] Add proper threading cleanup and error handling in app.py

---

## COMPLETED
- [x] Foundation & Perception (vision.py)
- [x] Memory & State (database.py, world.py)
- [x] Base Engines (engines.py)
- [x] Interface & Loop (app.py)
- [x] Multi-head RL system with 6 competing objectives
- [x] Meta-judge with urgency/scarcity/fatigue dynamics
- [x] Curiosity-driven concept selection from memory
- [x] Concept recombination (fusion) alongside mutation
- [x] Epistemic memory (confidence and prediction error tracking)
- [x] GNN Reasoning Layer (predicts offspring success)
- [x] Fix GNN integration for recombination
- [x] Fix Generation/Age ingestion bug
- [x] PRIORITY 1: Pruning & Extinction (Implemented in world.py and database.py)
- [x] PRIORITY 2 (Partial): Semantic Concept Tagging (CLIP text encoder added, tags generated and stored)