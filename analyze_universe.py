"""
Universe Analysis Tool
Diagnose if the model is actually learning semantic patterns.
"""

import sqlite3
import json
import numpy as np
from collections import defaultdict

def analyze_universe(db_path="universe.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("="*70)
    print("UNIVERSE SEMANTIC ANALYSIS")
    print("="*70)
    
    # Get all concepts
    cursor.execute("SELECT id, embedding, parent_ids, generation, coherence_score, novelty_score FROM concepts")
    concepts = cursor.fetchall()
    
    print(f"\nTotal Concepts: {len(concepts)}")
    
    # Analyze generations
    generations = [c[3] for c in concepts]
    print(f"Generation Range: {min(generations)} -> {max(generations)}")
    
    # Analyze parentage
    root_concepts = sum(1 for c in concepts if json.loads(c[2]) == [])
    derived_concepts = len(concepts) - root_concepts
    print(f"Root Concepts (real images): {root_concepts}")
    print(f"Derived Concepts (evolved): {derived_concepts}")
    
    # Analyze diversity (embedding variance)
    embeddings = []
    for c in concepts:
        emb = np.array(json.loads(c[1])).flatten()
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    
    # Pairwise distances (sample)
    sample_size = min(500, len(embeddings))
    sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
    sample_embs = embeddings[sample_idx]
    
    distances = []
    for i in range(len(sample_embs)):
        for j in range(i+1, min(i+50, len(sample_embs))):
            dist = np.linalg.norm(sample_embs[i] - sample_embs[j])
            distances.append(dist)
    
    avg_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    print(f"\nDiversity Metrics:")
    print(f"   Avg Distance: {avg_dist:.3f}")
    print(f"   Std Distance: {std_dist:.3f}")
    print(f"   Min Distance: {min(distances):.3f}")
    print(f"   Max Distance: {max(distances):.3f}")
    
    # Clustering analysis
    print(f"\nSemantic Clustering:")
    very_similar = sum(1 for d in distances if d < 0.1)
    similar = sum(1 for d in distances if 0.1 <= d < 0.5)
    different = sum(1 for d in distances if 0.5 <= d < 1.0)
    very_different = sum(1 for d in distances if d >= 1.0)
    
    total = len(distances)
    print(f"   Very Similar (<0.1):   {very_similar:5d} ({100*very_similar/total:5.1f}%)")
    print(f"   Similar (0.1-0.5):     {similar:5d} ({100*similar/total:5.1f}%)")
    print(f"   Different (0.5-1.0):   {different:5d} ({100*different/total:5.1f}%)")
    print(f"   Very Different (>1.0): {very_different:5d} ({100*very_different/total:5.1f}%)")
    
    # Evolution quality
    print(f"\nQuality Metrics:")
    coherences = []
    novelties = []
    for c in concepts:
        if c[4] is not None:
            try:
                coherences.append(float(c[4]))
            except (ValueError, TypeError):
                pass
        if c[5] is not None:
            try:
                novelties.append(float(c[5]))
            except (ValueError, TypeError):
                pass
    
    if coherences:
        print(f"   Avg Coherence: {np.mean(coherences):.3f}")
        print(f"   Avg Novelty:   {np.mean(novelties):.3f}")
    
    # Lineage depth analysis
    print(f"\nLineage Analysis:")
    lineage_depths = defaultdict(int)
    
    def get_depth(concept_id, depth=0, visited=None):
        if visited is None:
            visited = set()
        if concept_id in visited or depth > 100:  # Prevent infinite loops
            return depth
        visited.add(concept_id)
        
        cursor.execute("SELECT parent_ids FROM concepts WHERE id = ?", (concept_id,))
        row = cursor.fetchone()
        if not row:
            return depth
            
        parents = json.loads(row[0])
        if not parents:
            return depth
            
        max_parent_depth = 0
        for parent_id in parents:
            parent_depth = get_depth(parent_id, depth + 1, visited)
            max_parent_depth = max(max_parent_depth, parent_depth)
        return max_parent_depth
    
    # Sample lineage depths
    sample_concepts = np.random.choice([c[0] for c in concepts], min(100, len(concepts)), replace=False)
    depths = []
    for c_id in sample_concepts:
        depth = get_depth(c_id)
        depths.append(depth)
        lineage_depths[depth] += 1
    
    if depths:
        print(f"   Max Lineage Depth: {max(depths)}")
        print(f"   Avg Lineage Depth: {np.mean(depths):.1f}")
        print(f"   Depth Distribution:")
        for depth in sorted(lineage_depths.keys())[:10]:
            print(f"      Depth {depth:2d}: {lineage_depths[depth]:3d} concepts")
    
    # Diagnosis
    print(f"\nDIAGNOSIS:")
    
    issues = []
    if avg_dist < 0.3:
        issues.append("[!] LOW DIVERSITY: Concepts are too similar (avg dist < 0.3)")
    if very_similar / total > 0.5:
        issues.append("[!] CLUSTERING: >50% of concepts are very similar")
    if root_concepts < 100:
        issues.append("[!] LOW GROUNDING: Few real image anchors (<100)")
    if np.mean(novelties) < 0.1 if novelties else False:
        issues.append("[!] STAGNATION: Novelty scores are very low (<0.1)")
    if max(depths) < 3 if depths else False:
        issues.append("[!] SHALLOW EVOLUTION: No deep lineages (max depth < 3)")
    
    if not issues:
        print("   [OK] Universe appears healthy!")
    else:
        for issue in issues:
            print(f"   {issue}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if avg_dist < 0.3:
        print("   → Increase mutation rate (currently 0.15 → try 0.25)")
        print("   → Enable more recombination (30% → 50%)")
    if root_concepts < 500:
        print(f"   → Ingest more real images ({root_concepts}/2505 ingested)")
    if very_similar / total > 0.5:
        print("   → Run neuronal memory maintenance (consolidate duplicates)")
        print("   → Increase RL novelty reward weight")
    
    print("="*70)
    
    conn.close()

if __name__ == "__main__":
    analyze_universe()
