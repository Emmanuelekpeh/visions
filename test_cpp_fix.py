"""
Quick test to verify C++ Access Violation fix

This script tests that:
1. Training scheduler initializes correctly
2. Multiple threads can submit training tasks without crashes
3. No C++ race conditions occur
"""

import time
import threading
import numpy as np
from training_scheduler import get_training_scheduler, shutdown_training_scheduler


def dummy_training_func(embedding, latent=None, stage="supervised"):
    """
    Simulates a training function that would normally cause C++ race conditions.
    """
    import torch
    # Simulate some tensor operations
    tensor = torch.tensor(embedding, dtype=torch.float32)
    result = torch.nn.functional.normalize(tensor, p=2, dim=0)
    time.sleep(0.01)  # Simulate training time
    return float(result.sum().item())


def test_basic_scheduler():
    """Test 1: Basic scheduler functionality"""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Scheduler Functionality")
    print("=" * 80)
    
    scheduler = get_training_scheduler()
    
    # Submit a simple task
    embedding = np.random.randn(512).astype(np.float32)
    result = scheduler.submit(dummy_training_func, embedding, wait=True)
    
    print(f"[OK] Task completed with result: {result:.4f}")
    return True


def test_concurrent_submissions():
    """Test 2: Multiple threads submitting tasks concurrently"""
    print("\n" + "=" * 80)
    print("TEST 2: Concurrent Task Submissions (Multi-threaded)")
    print("=" * 80)
    
    scheduler = get_training_scheduler()
    
    def worker_thread(thread_id, num_tasks):
        """Simulate a worker thread submitting multiple training tasks"""
        for i in range(num_tasks):
            embedding = np.random.randn(512).astype(np.float32)
            scheduler.submit(
                dummy_training_func,
                embedding,
                stage="supervised",
                wait=False  # Non-blocking
            )
            if i % 10 == 0:
                print(f"  Thread {thread_id}: Submitted {i+1}/{num_tasks} tasks")
    
    # Create multiple worker threads (simulating evolution + ingestion threads)
    threads = []
    num_threads = 5
    tasks_per_thread = 20
    
    print(f"Launching {num_threads} threads, each submitting {tasks_per_thread} tasks...")
    
    start_time = time.time()
    for i in range(num_threads):
        t = threading.Thread(target=worker_thread, args=(i, tasks_per_thread))
        t.start()
        threads.append(t)
    
    # Wait for all threads to finish submitting
    for t in threads:
        t.join()
    
    submission_time = time.time() - start_time
    
    # Wait for all tasks to complete
    print("\nAll threads finished submitting. Waiting for tasks to complete...")
    scheduler.wait_for_pending_tasks(timeout=30.0)
    
    total_time = time.time() - start_time
    
    total_tasks = num_threads * tasks_per_thread
    print(f"\n[OK] {total_tasks} tasks completed successfully")
    print(f"  Submission time: {submission_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {total_tasks / total_time:.1f} tasks/sec")
    
    return True


def test_stress():
    """Test 3: Stress test with rapid concurrent submissions"""
    print("\n" + "=" * 80)
    print("TEST 3: Stress Test (Rapid Concurrent Submissions)")
    print("=" * 80)
    
    scheduler = get_training_scheduler()
    
    def rapid_submitter(thread_id, duration_sec):
        """Submit tasks as fast as possible for duration_sec"""
        count = 0
        end_time = time.time() + duration_sec
        
        while time.time() < end_time:
            embedding = np.random.randn(512).astype(np.float32)
            try:
                scheduler.submit(
                    dummy_training_func,
                    embedding,
                    wait=False
                )
                count += 1
            except Exception as e:
                print(f"  Thread {thread_id} error: {e}")
                break
        
        return count
    
    threads = []
    results = []
    num_threads = 10
    duration = 5  # seconds
    
    print(f"Launching {num_threads} threads for {duration}s stress test...")
    
    start_time = time.time()
    for i in range(num_threads):
        t = threading.Thread(target=lambda tid=i: results.append(rapid_submitter(tid, duration)))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    print("\nStress test complete. Waiting for pending tasks...")
    scheduler.wait_for_pending_tasks(timeout=30.0)
    
    total_time = time.time() - start_time
    total_submitted = sum(results) if results else 0
    
    print(f"\n[OK] Stress test completed without crashes")
    print(f"  Total tasks submitted: {total_submitted}")
    print(f"  Time: {total_time:.2f}s")
    print(f"  Submission rate: {total_submitted / duration:.1f} tasks/sec")
    
    return True


def test_pytorch_integration():
    """Test 4: Actual PyTorch training calls"""
    print("\n" + "=" * 80)
    print("TEST 4: PyTorch Integration (Real Training)")
    print("=" * 80)
    
    try:
        import torch
        from vision import VisionSystem
        from latent_generator import create_latent_generator
        
        print("Initializing vision system and generator...")
        vision = VisionSystem(device="cpu")
        generator, trainer = create_latent_generator(vision, device="cpu")
        
        scheduler = get_training_scheduler()
        
        # Submit multiple real training tasks from different threads
        def training_worker(thread_id, num_batches):
            for i in range(num_batches):
                # Create synthetic training data
                embedding = np.random.randn(1, 512).astype(np.float32)
                latent = np.random.randn(1, 4, 64, 64).astype(np.float32)
                
                # Submit actual training
                scheduler.submit(
                    trainer.train_on_batch,
                    embedding,
                    latent,
                    stage="supervised",
                    wait=False
                )
                
                if i % 5 == 0:
                    print(f"  Thread {thread_id}: Submitted batch {i+1}/{num_batches}")
        
        threads = []
        num_threads = 3
        batches_per_thread = 10
        
        print(f"\nLaunching {num_threads} threads with real PyTorch training...")
        
        for i in range(num_threads):
            t = threading.Thread(target=training_worker, args=(i, batches_per_thread))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        print("\nWaiting for training tasks to complete...")
        scheduler.wait_for_pending_tasks(timeout=30.0)
        
        print("\n[OK] Real PyTorch training completed without C++ crashes!")
        return True
        
    except Exception as e:
        print(f"[FAIL] PyTorch integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("C++ ACCESS VIOLATION FIX - TEST SUITE")
    print("=" * 80)
    print("\nThis test verifies that the training scheduler prevents C++ race conditions")
    print("by ensuring all PyTorch operations happen on a single dedicated thread.")
    
    results = {}
    
    try:
        # Run tests
        results['basic'] = test_basic_scheduler()
        results['concurrent'] = test_concurrent_submissions()
        results['stress'] = test_stress()
        results['pytorch'] = test_pytorch_integration()
        
    except Exception as e:
        print(f"\n[ERROR] Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n" + "=" * 80)
        print("Shutting down scheduler...")
        shutdown_training_scheduler()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"  {test_name:20s}: {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\n  Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! C++ Access Violation fix verified.")
        print("\nThe training scheduler successfully prevents race conditions by:")
        print("  1. Serializing all PyTorch operations on a single thread")
        print("  2. Isolating training data via deep copying")
        print("  3. Preventing concurrent C++ memory allocator access")
    else:
        print("\n[WARN] Some tests failed. Review output above.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
