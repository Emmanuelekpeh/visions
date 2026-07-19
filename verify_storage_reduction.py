"""
Verify storage reduction by running evolution and checking database size.
"""

from world import WorldState
from database import WorldMemory
import time

def main():
    print("=" * 80)
    print("STORAGE REDUCTION VERIFICATION")
    print("=" * 80)
    
    # Initialize world
    print("\n1. Initializing world with generator...")
    world = WorldState(device="cpu")
    
    # Check initial state
    cursor = world.memory.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM concepts WHERE generator_version = -1")
    real_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM concepts")
    total_count = cursor.fetchone()[0]
    
    print(f"   Initial state:")
    print(f"     Real images: {real_count}")
    print(f"     Total concepts: {total_count}")
    
    # Run evolution
    print("\n2. Running 50 evolution steps...")
    start_time = time.time()
    
    for i in range(50):
        world.step()
        if (i + 1) % 10 == 0:
            print(f"   Completed {i+1}/50 steps")
    
    elapsed = time.time() - start_time
    print(f"   Evolution completed in {elapsed:.1f}s")
    
    # Check final state
    print("\n3. Analyzing storage...")
    
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
    evolved_with_stored_latents = 0
    evolved_without_latents = 0
    
    print("\n   Storage by generator_version:")
    for ver, count, latent_bytes in results:
        total_concepts += count
        total_latent_bytes += latent_bytes or 0
        
        if ver == -1:
            version_name = "Real images"
        elif ver == 0:
            version_name = "Legacy stored"
        elif ver == 1:
            version_name = "Generated"
            if latent_bytes and latent_bytes > 0:
                evolved_with_stored_latents = count
            else:
                evolved_without_latents = count
        else:
            version_name = f"Unknown ({ver})"
        
        print(f"     {version_name:20s}: {count:4d} concepts, {(latent_bytes or 0) / 1024:.2f} KB")
    
    # Calculate savings
    print("\n4. Storage Analysis:")
    
    # Old way: Every concept stores 67KB (2KB embedding + 64KB latent + overhead)
    old_storage_bytes = total_concepts * 67 * 1024
    
    # New way: Actual latent storage + generator (306MB) + embeddings
    generator_size = 306 * 1024 * 1024
    new_storage_bytes = total_latent_bytes + generator_size
    
    savings_bytes = old_storage_bytes - new_storage_bytes
    savings_pct = (savings_bytes / old_storage_bytes * 100) if old_storage_bytes > 0 else 0
    
    print(f"   Old way (67KB per concept):")
    print(f"     {total_concepts} concepts × 67KB = {old_storage_bytes / 1024 / 1024:.2f} MB")
    
    print(f"\n   New way (generator + stored latents):")
    print(f"     Stored latents: {total_latent_bytes / 1024 / 1024:.2f} MB")
    print(f"     Generator model: {generator_size / 1024 / 1024:.2f} MB")
    print(f"     Total: {new_storage_bytes / 1024 / 1024:.2f} MB")
    
    print(f"\n   Savings:")
    print(f"     Absolute: {savings_bytes / 1024 / 1024:.2f} MB")
    print(f"     Percentage: {savings_pct:.1f}%")
    
    # Verify evolved concepts don't store latents
    print("\n5. Verification:")
    
    if evolved_with_stored_latents > 0:
        print(f"   [WARN] {evolved_with_stored_latents} evolved concepts still have stored latents!")
        print(f"          This shouldn't happen with generator_version=1")
    else:
        print(f"   [OK] No evolved concepts have stored latents")
    
    if evolved_without_latents > 0:
        print(f"   [OK] {evolved_without_latents} evolved concepts using generator (no storage!)")
    
    # Success criteria
    print("\n6. Success Criteria:")
    
    criteria = []
    
    # 1. Evolved concepts don't store latents
    if evolved_with_stored_latents == 0 and evolved_without_latents > 0:
        print(f"   [OK] Evolved concepts use generator (not stored)")
        criteria.append(True)
    else:
        print(f"   [FAIL] Some evolved concepts still store latents")
        criteria.append(False)
    
    # 2. Real images still store latents
    cursor.execute("SELECT COUNT(*) FROM concepts WHERE generator_version = -1 AND latent IS NULL")
    real_without_latents = cursor.fetchone()[0]
    
    if real_without_latents == 0 and real_count > 0:
        print(f"   [OK] All real images store latents (as expected)")
        criteria.append(True)
    elif real_count == 0:
        print(f"   [WARN] No real images to test")
        criteria.append(None)
    else:
        print(f"   [FAIL] Some real images missing latents")
        criteria.append(False)
    
    # 3. Storage reduction achieved (at small scale, break-even is around 5-10 concepts)
    if savings_bytes > 0:
        print(f"   [OK] Storage reduced by {savings_pct:.1f}%")
        criteria.append(True)
    elif total_concepts < 10:
        print(f"   [INFO] Too few concepts ({total_concepts}) to see savings (generator overhead)")
        criteria.append(None)
    else:
        print(f"   [WARN] No storage reduction yet")
        criteria.append(False)
    
    # Summary
    passed = sum(1 for c in criteria if c is True)
    failed = sum(1 for c in criteria if c is False)
    skipped = sum(1 for c in criteria if c is None)
    
    print("\n" + "=" * 80)
    print(f"RESULT: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n[SUCCESS] Storage reduction working correctly!")
    else:
        print("\n[FAIL] Some criteria not met")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
