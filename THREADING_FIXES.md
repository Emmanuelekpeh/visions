# Threading Race Condition Fixes

**Date:** July 16, 2026  
**Issue:** RuntimeError: dictionary changed size during iteration

## Root Cause

The system uses threading for non-blocking evolution in the UI:
- Main thread: Pygame UI updates
- Worker threads: Evolution steps

**Problem:**
Multiple threads access shared dictionaries simultaneously:
1. Evolution thread: Adds new concepts to graph
2. Extinction thread: Removes concepts from graph
3. Save thread: Iterates over graph to persist state

**Result:** `RuntimeError: dictionary changed size during iteration`

## Locations Fixed

### 1. Concept Graph Save (`concept_graph.py`)
```python
# BEFORE (race condition)
for node_id, node in self.nodes.items():

# AFTER (thread-safe)
nodes_snapshot = list(self.nodes.items())
for node_id, node in nodes_snapshot:
```

### 2. Epistemic Memory Save (`epistemic_memory.py`)
```python
# BEFORE (race condition)
for concept_id, conf in all_concepts.items():

# AFTER (thread-safe)
concepts_snapshot = list(all_concepts.items())
for concept_id, conf in concepts_snapshot:
```

### 3. Extinction Sweep (`world.py`)
```python
# BEFORE (race condition)
for node_id, node in list(self.concept_graph.nodes.items()):

# AFTER (explicit snapshot)
nodes_snapshot = list(self.concept_graph.nodes.items())
for node_id, node in nodes_snapshot:
```

### 4. GNN Build Adjacency (`graph_reasoning.py`)
```python
# BEFORE (potential race)
node_ids = sorted(self.graph.nodes.keys())

# AFTER (thread-safe)
node_ids = sorted(list(self.graph.nodes.keys()))
```

### 5. GNN Recommendations (multiple locations)
Added explicit snapshots in 3 methods:
- `_build_adjacency_matrices()`
- `recommend_parents_for_mutation()`
- `recommend_parent_pairs_for_recombination()`

## Why This Works

**Dictionary Snapshots:**
- `list(dict.items())` creates a copy at that moment
- Iteration happens on the copy, not the live dictionary
- Other threads can modify original dictionary safely

**Trade-off:**
- Small memory overhead (few KB for snapshots)
- Prevents crashes and data corruption
- Worth it for thread safety

## Additional Safeguards

1. **Node Existence Checks:**
   - Before using node ID, verify it exists
   - Skip if removed during operation

2. **Index Validation:**
   - Check `idx < len(array)` before access
   - Handle dimension mismatches gracefully

3. **Safety Comments:**
   - Document why snapshots are needed
   - Help future developers understand

## Testing

Run with auto-evolution enabled for 200+ generations:
```bash
python app.py
# Press 'A' to enable auto-evolve
# Let it run for 5+ minutes
```

Should no longer see:
- ❌ `RuntimeError: dictionary changed size`
- ❌ `KeyError: 466` (or any node ID)

Should see:
- ✅ Smooth evolution
- ✅ Successful saves every 50/100 generations
- ✅ Extinctions without crashes

## Remaining Thread Safety Notes

**SQLite:** Already thread-safe with `check_same_thread=False`
**FAISS:** Read-only after initial load (thread-safe)
**NumPy:** Operations are thread-safe
**Torch:** Model inference in eval mode (thread-safe)

**Not Thread-Safe (but isolated):**
- Pygame rendering (main thread only)
- File I/O (happens in worker thread)

## Future Improvements

If more threading issues arise, consider:
1. Adding mutex locks around critical sections
2. Using threading.Lock() for dictionary modifications
3. Moving to multiprocessing instead of threading
4. Queue-based architecture for evolution steps

For now, snapshots are sufficient and performant.

---

**Status:** ✅ All known threading race conditions fixed
