# Architecture Clarification: Where the Latent Generator Sits

## Quick Answer

**Latent Generator** and **GNN** are at the **same level** but serve **different purposes**:

- **GNN**: Recommends **which concepts** to evolve (navigation/selection)
- **Latent Generator**: Creates **latent vectors** from embeddings (reconstruction)

They're **complementary**, not hierarchical.

---

## The Complete Architecture Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    USER / PYGAME UI                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      WorldState                             │
│  (Orchestrates everything)                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌──────────────────────┐              ┌──────────────────────┐
│   SELECTION LAYER    │              │  GENERATION LAYER    │
│  (What to evolve)    │              │  (How to create it)  │
├──────────────────────┤              ├──────────────────────┤
│ • GNN                │              │ • Latent Generator   │
│ • Epistemic Memory   │              │ • DreamEngine        │
│ • Meta-Controller    │              │   (mutation/recom)   │
│ • Concept Graph      │              │ • Vision (VAE)       │
└──────────────────────┘              └──────────────────────┘
        ↓                                       ↓
        └───────────────────┬───────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION LAYER                          │
│  • Multi-head RL (6 reward heads)                          │
│  • Physics Engine (transformations)                         │
│  • Ecology Engine (species/culture)                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER                            │
│  • Database (SQLite)                                        │
│  • FAISS Index                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## When Each Component Is Used

### During Ingestion (Real Images)

```
1. Load image from dataset
   ↓
2. Vision System (CLIP + VAE)
   - CLIP extracts embedding (512-d) [stored]
   - VAE encodes to latent (4×64×64) [stored]
   ↓
3. Database stores both
   - generator_version = -1 (real image)
   ↓
4. Concept Graph adds as Reality node
   ↓
5. Epistemic Memory adds as Landmark (confidence=1.0)

❌ Latent Generator NOT used (real images keep original latents)
❌ GNN NOT used (no evolution during ingestion)
```

### During Evolution (Creating New Concepts)

```
1. SELECTION PHASE (GNN's job)
   ↓
   GNN recommends: "These parent concepts are promising"
   - Based on learned patterns
   - Based on combination success history
   - Based on exploration value
   ↓
   Meta-Controller adjusts: "Weight novelty higher now"
   ↓
   Epistemic Memory filters: "These have high confidence"
   ↓
   Result: Selected parent concept(s)

2. GENERATION PHASE (Latent Generator's job)
   ↓
   Get parent latent:
   - If real image: Load from database
   - If evolved: Latent Generator creates from embedding
   ↓
   DreamEngine mutates/recombines latent
   ↓
   Extract new embedding from mutated latent
   ↓
   Store ONLY embedding (not latent!)
   - generator_version = 1 (will generate on-demand)
   ↓
   ✅ Latent Generator trains on this example (incremental learning)

3. EVALUATION PHASE
   ↓
   Multi-head RL evaluates concept
   ↓
   Physics Engine tracks transformation success
   ↓
   Ecology Engine classifies into species

4. LATER RETRIEVAL (Latent Generator's job again)
   ↓
   Need to view/evolve this concept:
   - Load embedding from database (2KB)
   - Latent Generator: embedding → latent (0.5ms)
   - Vision VAE: latent → image
```

---

## Latent Generator Learning Schedule

### Stage 1: Initial Training (What you already did)
```bash
python train_latent_generator.py --from-images --limit 50 --epochs 50
```

Learned from 50 real images:
- Embedding → Latent mapping for dataset
- Result: 78% cosine similarity (working but improvable)

### Stage 2: Incremental Learning (Happens automatically during evolution)

In `world.py` after accepting a new evolved concept:
```python
if not is_real_image:
    generator_version = 1
    latent_to_store = None  # Don't store!
    
    # Train generator on this new concept (incremental learning)
    self.latent_generator_trainer.train_on_batch(
        embedding_np.reshape(1, -1),
        latent.cpu().numpy().reshape(1, 4, 64, 64),
        stage="supervised"
    )
```

**What this does**:
- Generator learns the evolved latent space
- Adapts to mutations/recombinations
- Continuously improves over generations

### Stage 3: Full Training (Optional, recommended)
```bash
python train_latent_generator.py --from-images --epochs 200 --batch-size 8
```

Train on all 242 ingested images:
- Better quality (target: 90%+ similarity)
- More robust generation
- Better generalization

---

## How GNN and Latent Generator Work Together

```
Evolution Step Example:

1. GNN: "I recommend combining concept #42 (forest) and #91 (glass)"
   - Based on: Previous forest+glass → 0.91 success rate
   - Based on: This region has high exploration value
   ↓
2. Latent Generator: "Let me get those latents"
   - concept #42: embedding → generate latent
   - concept #91: embedding → generate latent
   ↓
3. DreamEngine: "I'll recombine them"
   - latent_42 * 0.5 + latent_91 * 0.5 → new_latent
   ↓
4. Vision System: "Here's what it looks like"
   - new_latent → decode → glass_forest image
   ↓
5. Multi-head RL: "This scores 0.82!"
   ↓
6. GNN updates: "Forest+glass worked again! Recommend more"
7. Latent Generator trains: "I learned this new latent"
```

**They complement each other**:
- GNN learns **navigation** (where to explore)
- Latent Generator learns **reconstruction** (how to get there)

---

## Why Two Separate Systems?

### GNN (Graph Neural Network)
**Purpose**: Recommendation engine for evolution
- Input: Graph structure (nodes + edges)
- Output: "These concepts/pairs will produce good offspring"
- Learns: Patterns in combination success
- Storage: ~10MB (lightweight)

### Latent Generator
**Purpose**: Latent space generator
- Input: Concept embedding (512-d)
- Output: VAE latent (4×64×64)
- Learns: Embedding → Latent mapping
- Storage: ~306MB (heavyweight)

**Why not combine them?**
1. Different tasks (navigation vs reconstruction)
2. Different inputs (graph vs embedding)
3. Different scales (10MB vs 306MB)
4. GNN could work with stored latents (but storage was the problem!)

---

## Alternative: What If GNN Generated Latents?

You could theoretically have the GNN output latents directly:

```python
# Hypothetical combined system
class CombinedGNN(nn.Module):
    def forward(self, graph):
        # Process graph structure
        node_features = graph_convolution(graph)
        
        # For each node, predict its latent
        latents = latent_head(node_features)  # [N, 4, 64, 64]
        
        return latents
```

**Why we didn't do this**:
1. **GNN needs stored latents to learn** - Chicken-and-egg problem
2. **GNN is for graph reasoning** - Not designed for dense generation
3. **Separate concerns** - GNN recommends, Generator creates
4. **Generator is simpler** - Just embedding → latent, easier to train

---

## The Division of Labor

```
┌──────────────────────────────────────────────────────────┐
│                    CONCEPT LIFECYCLE                     │
└──────────────────────────────────────────────────────────┘

1. BIRTH (Real image ingestion)
   Vision: Extract embedding + latent → Store both
   
2. SELECTION (Evolution)
   GNN: "Evolve these concepts"
   Epistemic Memory: "These are high confidence"
   Meta-Controller: "Weight these objectives"
   
3. GENERATION (Create offspring)
   Latent Generator: Get parent latents (generate if needed)
   DreamEngine: Mutate/recombine
   Vision: Decode to image
   
4. EVALUATION (Accept/reject)
   Multi-head RL: Score the concept
   Physics: Track transformation
   Ecology: Assign species
   
5. STORAGE (If accepted)
   Database: Store embedding only (generator_version=1)
   Latent Generator: Train on this example
   GNN: Record success for future recommendations
   
6. RETRIEVAL (Later use)
   Latent Generator: Regenerate latent on-demand
   Vision: Decode to image
```

---

## Training Data Flow

### GNN Training Data
```
Source: Evolution outcomes
Data: (parent_a, parent_b, offspring_quality)

Example:
- forest + glass → 0.91 success
- forest + metal → 0.76 success
- water + fire → 0.12 failure

GNN learns: "Organic + inorganic works well"
```

### Latent Generator Training Data
```
Source 1: Real images (initial training)
Data: (embedding, latent) pairs from dataset

Source 2: Evolved concepts (incremental)
Data: (embedding, mutated_latent) pairs from evolution

Generator learns: "This embedding space maps to this latent space"
```

**They use different training data for different purposes!**

---

## Memory Footprint

```
System Component         RAM Usage    Disk Usage
──────────────────────  ──────────  ────────────
Vision (CLIP + VAE)     ~500 MB     ~400 MB
Latent Generator        ~306 MB     ~306 MB
GNN                     ~10 MB      ~10 MB
Concept Graph           ~50 MB      ~5 MB
Epistemic Memory        ~20 MB      ~2 MB
Multi-head RL           ~1 MB       0 MB
Database (per concept)  0 MB        ~2 KB (was 67KB!)
──────────────────────  ──────────  ────────────
Total System            ~900 MB     ~725 MB + 2KB per concept
```

**Before latent generator**: 67KB per concept (unbounded)  
**After latent generator**: 2KB per concept (bounded)

---

## Summary

### Where Latent Generator Sits
- **Level**: Generation layer (parallel to GNN)
- **Purpose**: Reconstruct latents from embeddings
- **When used**: Every time an evolved concept's latent is needed
- **When trained**: 
  1. Initial: On real images (you did this)
  2. Incremental: On each evolved concept (automatic)
  3. Full: On complete dataset (optional, recommended)

### Relationship to GNN
- **Independent systems** with different jobs
- **Complementary**: GNN navigates, Generator reconstructs
- **Both learn**: From evolution outcomes, but different things
- **Not hierarchical**: Neither depends on the other

### Key Insight
The latent generator **replaces storage**, not other components:
- **Before**: Store every latent (64KB each)
- **After**: Generate from embedding (2KB each)

It's a **storage optimization**, not an architectural change.

---

## Visual: Data Flow During Evolution

```
┌─────────────┐
│     GNN     │ "Recommend forest + glass"
└──────┬──────┘
       ↓
┌──────────────────────────────────────┐
│  Get Latents (Generator's job)      │
│  - forest: embedding → latent (gen)  │
│  - glass:  embedding → latent (gen)  │
└──────┬───────────────────────────────┘
       ↓
┌─────────────┐
│DreamEngine  │ Recombine latents
└──────┬──────┘
       ↓
┌─────────────┐
│   Vision    │ Decode → glass_forest image
└──────┬──────┘
       ↓
┌─────────────┐
│Multi-head RL│ Evaluate → 0.82 score
└──────┬──────┘
       ↓
       ├───→ GNN: "Record success" (learns navigation)
       │
       └───→ Generator: "Train on this" (learns reconstruction)
```

**Two learning loops, same evolution!**

---

Does this clarify where the latent generator sits and how it relates to the GNN?
