"""
Dataset Ingestion Script
Ingests all images from dataset/ folder as anchor concepts.
This grounds the universe in real-world visual data.
"""

import os
import time
from world import WorldState

def main():
    print("=" * 80)
    print("DATASET INGESTION: Grounding Universe in Reality")
    print("=" * 80)
    
    # Check dataset
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"ERROR: {dataset_dir} folder not found")
        return
    
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    all_files = [f for f in os.listdir(dataset_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    print(f"\nFound {len(all_files)} images in dataset/")
    
    # Initialize world
    print("\nInitializing universe (this may take a moment)...")
    world = WorldState(device="cpu")
    
    # Check current state
    cursor = world.memory.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM concepts")
    existing_concepts = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM ingested_files")
    already_ingested = cursor.fetchone()[0]
    
    print(f"\nCurrent Universe State:")
    print(f"  Existing concepts: {existing_concepts}")
    print(f"  Already ingested: {already_ingested}")
    print(f"  To ingest: {len(all_files) - already_ingested}")
    
    if already_ingested >= len(all_files):
        print("\n✓ All images already ingested!")
        return
    
    # Confirm
    print(f"\nThis will ingest ~{len(all_files) - already_ingested} new images.")
    print("Estimated time: ~10-30 seconds per image on CPU")
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Ingest
    print("\n" + "=" * 80)
    print("INGESTION STARTED")
    print("=" * 80)
    
    start_time = time.time()
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, filename in enumerate(all_files):
        # Check if already ingested
        if world.memory.is_file_ingested(filename):
            skip_count += 1
            print(f"[{i+1}/{len(all_files)}] SKIP: {filename} (already ingested)")
            continue
        
        filepath = os.path.join(dataset_dir, filename)
        print(f"\n[{i+1}/{len(all_files)}] Processing: {filename}")
        
        try:
            success, c_id = world.ingest_image(filepath)
            if success and c_id is not None:
                world.memory.mark_file_ingested(filename, c_id)
                success_count += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / (success_count + error_count) if (success_count + error_count) > 0 else 0
                remaining = avg_time * (len(all_files) - i - 1)
                print(f"  ✓ Concept ID: {c_id}")
                print(f"  ⏱ Elapsed: {elapsed:.1f}s | Avg: {avg_time:.1f}s/img | ETA: {remaining/60:.1f}min")
            else:
                error_count += 1
                print(f"  ✗ Failed to ingest")
        except Exception as e:
            error_count += 1
            print(f"  ✗ ERROR: {e}")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  Successfully ingested: {success_count}")
    print(f"  Skipped (duplicate): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total time: {elapsed/60:.2f} minutes")
    print(f"  Average: {elapsed/success_count:.2f} seconds per image" if success_count > 0 else "")
    
    # Final state
    cursor.execute("SELECT COUNT(*) FROM concepts")
    final_concepts = cursor.fetchone()[0]
    print(f"\nFinal Universe State:")
    print(f"  Total concepts: {final_concepts}")
    print(f"  Anchor concepts (real images): {success_count + skip_count}")
    print(f"  Synthetic concepts: {existing_concepts}")
    print(f"  Reality anchored: {100*(success_count+skip_count)/final_concepts:.1f}%")
    
    print("\n✓ Universe is now grounded in reality!")
    print("=" * 80)

if __name__ == "__main__":
    main()
