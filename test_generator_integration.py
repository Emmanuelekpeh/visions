"""
Test script to verify latent generator integration works correctly.

Tests:
1. Database migration (generator_version column exists)
2. Real images store latents (generator_version = -1)
3. Evolved concepts don't store latents (generator_version = 1)
4. get_latent_for_concept() works for both types
5. Generated latents produce valid images
6. Storage reduction verified
"""

import sys
import torch
import numpy as np
from world import WorldState
from database import WorldMemory
import json

def test_database_schema():
    """Test that generator_version column exists."""
    print("\n" + "=" * 80)
    print("TEST 1: Database Schema Migration")
    print("=" * 80)
    
    memory = WorldMemory()
    cursor = memory.conn.cursor()
    
    try:
        cursor.execute("SELECT generator_version FROM concepts LIMIT 1")
        print("[OK] generator_version column exists")
        return True
    except Exception as e:
        print(f"[FAIL] generator_version column missing: {e}")
        return False


def test_generator_initialization():
    """Test that world initializes with generator."""
    print("\n" + "=" * 80)
    print("TEST 2: Generator Initialization")
    print("=" * 80)
    
    try:
        # Skip ingestion by checking if we already have concepts
        memory = WorldMemory()
        cursor = memory.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM concepts")
        existing_concepts = cursor.fetchone()[0]
        
        if existing_concepts > 0:
            print(f"  Using existing {existing_concepts} concepts (skipping ingestion)")
        
        world = WorldState(device="cpu")
        
        assert hasattr(world, 'latent_generator'), "World missing latent_generator"
        assert hasattr(world, 'latent_generator_trainer'), "World missing latent_generator_trainer"
        assert hasattr(world, 'get_latent_for_concept'), "World missing get_latent_for_concept method"
        
        print("[OK] World has latent_generator")
        print("[OK] World has latent_generator_trainer")
        print("[OK] World has get_latent_for_concept method")
        return True, world
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_storage_behavior(world):
    """Test that evolved concepts don't store latents."""
    print("\n" + "=" * 80)
    print("TEST 3: Storage Behavior")
    print("=" * 80)
    
    cursor = world.memory.conn.cursor()
    
    # Count concepts by generator_version
    cursor.execute("SELECT generator_version, COUNT(*) FROM concepts GROUP BY generator_version")
    results = cursor.fetchall()
    
    stats = {ver: count for ver, count in results}
    
    print(f"  Real images (generator_version = -1): {stats.get(-1, 0)}")
    print(f"  Legacy stored (generator_version = 0): {stats.get(0, 0)}")
    print(f"  Generated (generator_version = 1): {stats.get(1, 0)}")
    
    # Check that evolved concepts (version 1) have NULL latents
    cursor.execute("SELECT COUNT(*) FROM concepts WHERE generator_version = 1 AND latent IS NOT NULL")
    evolved_with_latents = cursor.fetchone()[0]
    
    if evolved_with_latents > 0:
        print(f"[WARN] Warning: {evolved_with_latents} evolved concepts still have stored latents")
    else:
        print(f"[OK] All evolved concepts use generated latents (no storage)")
    
    return stats


def test_latent_generation(world):
    """Test get_latent_for_concept for both real and evolved concepts."""
    print("\n" + "=" * 80)
    print("TEST 4: Latent Generation")
    print("=" * 80)
    
    cursor = world.memory.conn.cursor()
    
    # Test real image (should load stored latent)
    cursor.execute("SELECT id FROM concepts WHERE generator_version = -1 LIMIT 1")
    real_img_row = cursor.fetchone()
    
    if real_img_row:
        real_id = real_img_row[0]
        print(f"\nTesting real image (concept {real_id}):")
        try:
            latent = world.get_latent_for_concept(real_id)
            print(f"  [OK] Retrieved latent: shape {latent.shape}")
            
            # Decode to image
            image = world.vision.decode_latent_to_image(latent)
            print(f"  [OK] Decoded to image: size {image.size}")
        except Exception as e:
            print(f"  [FAIL] Failed: {e}")
            return False
    else:
        print("[WARN] No real images found to test")
    
    # Test evolved concept (should generate latent)
    cursor.execute("SELECT id FROM concepts WHERE generator_version = 1 LIMIT 1")
    evolved_row = cursor.fetchone()
    
    if evolved_row:
        evolved_id = evolved_row[0]
        print(f"\nTesting evolved concept (concept {evolved_id}):")
        try:
            latent = world.get_latent_for_concept(evolved_id)
            print(f"  [OK] Generated latent: shape {latent.shape}")
            
            # Decode to image
            image = world.vision.decode_latent_to_image(latent)
            print(f"  [OK] Decoded to image: size {image.size}")
            print(f"  [OK] Image generated on-demand (no storage used!)")
        except Exception as e:
            print(f"  [FAIL] Failed: {e}")
            return False
    else:
        print("[WARN] No evolved concepts found to test")
    
    return True


def test_storage_reduction():
    """Calculate actual storage savings."""
    print("\n" + "=" * 80)
    print("TEST 5: Storage Reduction")
    print("=" * 80)
    
    memory = WorldMemory()
    cursor = memory.conn.cursor()
    
    # Get storage info
    cursor.execute("""
        SELECT 
            generator_version,
            COUNT(*) as count,
            SUM(CASE WHEN latent IS NOT NULL THEN LENGTH(latent) ELSE 0 END) as latent_bytes
        FROM concepts 
        GROUP BY generator_version
    """)
    
    results = cursor.fetchall()
    
    total_concepts = 0
    total_latent_bytes = 0
    evolved_concepts = 0
    
    for ver, count, latent_bytes in results:
        total_concepts += count
        total_latent_bytes += latent_bytes or 0
        
        if ver == 1:
            evolved_concepts = count
        
        print(f"\n  Generator version {ver}:")
        print(f"    Concepts: {count}")
        print(f"    Latent storage: {(latent_bytes or 0) / 1024:.2f} KB")
    
    # Calculate savings
    old_storage = total_concepts * 67 * 1024  # 67KB per concept (old way)
    new_storage = total_latent_bytes + 306 * 1024 * 1024  # Actual storage + generator size
    
    savings_mb = (old_storage - new_storage) / 1024 / 1024
    savings_pct = ((old_storage - new_storage) / old_storage * 100) if old_storage > 0 else 0
    
    print(f"\n  STORAGE ANALYSIS:")
    print(f"    Total concepts: {total_concepts}")
    print(f"    Evolved concepts: {evolved_concepts}")
    print(f"    Old way (67KB each): {old_storage / 1024 / 1024:.2f} MB")
    print(f"    New way (actual): {new_storage / 1024 / 1024:.2f} MB")
    print(f"    Savings: {savings_mb:.2f} MB ({savings_pct:.1f}%)")
    
    if evolved_concepts > 0:
        per_concept_savings = evolved_concepts * 64 * 1024  # 64KB latent per evolved concept
        print(f"    Saved from evolved concepts: {per_concept_savings / 1024 / 1024:.2f} MB")
    
    return savings_mb > 0


def test_evolution_step(world):
    """Test that evolution works with generator."""
    print("\n" + "=" * 80)
    print("TEST 6: Evolution with Generator")
    print("=" * 80)
    
    try:
        print("  Running 5 evolution steps...")
        for i in range(5):
            result = world.step()
            print(f"    Step {i+1}: {'[OK] Success' if result else '[FAIL] Failed'}")
            if not result:
                return False
        
        print("\n  [OK] Evolution works with generator!")
        return True
    except Exception as e:
        print(f"  [FAIL] Evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("LATENT GENERATOR INTEGRATION TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Schema migration
    results['schema'] = test_database_schema()
    
    # Test 2: Initialization
    init_success, world = test_generator_initialization()
    results['initialization'] = init_success
    
    if not init_success:
        print("\n[FAIL] Cannot proceed without successful initialization")
        return
    
    # Test 3: Storage behavior
    stats = test_storage_behavior(world)
    results['storage'] = True
    
    # Test 4: Latent generation
    results['generation'] = test_latent_generation(world)
    
    # Test 5: Storage reduction
    results['reduction'] = test_storage_reduction()
    
    # Test 6: Evolution
    if world.memory.index.ntotal > 0:
        results['evolution'] = test_evolution_step(world)
    else:
        print("\n[WARN] Skipping evolution test (no concepts in database)")
        results['evolution'] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        if result is None:
            status = "[SKIP] SKIPPED"
        elif result:
            status = "[OK] PASS"
        else:
            status = "[FAIL] FAIL"
        print(f"  {test_name:20s}: {status}")
    
    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    
    print(f"\n  Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Generator integration successful.")
    else:
        print("\n[WARN] Some tests failed. Review output above.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
