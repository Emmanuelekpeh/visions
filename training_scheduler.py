"""
Thread-Safe Training Scheduler

Solves C++ Access Violation in PyTorch's Adam optimizer by ensuring all
PyTorch training operations happen on a single dedicated thread.

Problem:
    - PyTorch's C++ backend is NOT thread-safe
    - Adam optimizer maintains momentum buffers in C++ memory
    - Python locks (RLock) don't protect C++ memory allocator operations
    - Multiple threads calling optimizer.step() → C++ race condition → Access Violation

Solution:
    - Single dedicated training thread with a queue
    - All training requests are serialized through this thread
    - PyTorch operations never overlap at C++ level
"""

import threading
import queue
from typing import Callable, Any, Optional
from dataclasses import dataclass
import traceback


@dataclass
class TrainingTask:
    """A training task to be executed on the training thread."""
    func: Callable
    args: tuple
    kwargs: dict
    result_queue: Optional[queue.Queue] = None


class TrainingScheduler:
    """
    Single-threaded training scheduler that ensures all PyTorch operations
    happen on one dedicated thread, preventing C++ race conditions.
    """
    
    def __init__(self):
        self.task_queue = queue.Queue(maxsize=100)
        self.shutdown_flag = threading.Event()
        self.worker_thread = None
        self._start_worker()
    
    def _start_worker(self):
        """Start the dedicated training worker thread."""
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="PyTorchTrainingWorker"
        )
        self.worker_thread.start()
    
    def _worker_loop(self):
        """
        Main loop for the training worker thread.
        All PyTorch training operations execute here sequentially.
        """
        while not self.shutdown_flag.is_set():
            try:
                # Wait for next task with timeout to check shutdown flag
                task = self.task_queue.get(timeout=0.1)
                
                try:
                    # Execute the training task
                    result = task.func(*task.args, **task.kwargs)
                    
                    # Return result if requested
                    if task.result_queue is not None:
                        task.result_queue.put(("success", result))
                        
                except Exception as e:
                    # Capture exception and return to caller
                    error_msg = f"Training task failed: {e}\n{traceback.format_exc()}"
                    print(f"[TrainingScheduler] {error_msg}")
                    
                    if task.result_queue is not None:
                        task.result_queue.put(("error", error_msg))
                
                finally:
                    self.task_queue.task_done()
                    
            except queue.Empty:
                # No tasks, continue checking shutdown flag
                continue
    
    def submit(self, func: Callable, *args, wait: bool = False, **kwargs) -> Any:
        """
        Submit a training task to the scheduler.
        
        Args:
            func: Function to execute (e.g., latent_generator_trainer.train_on_batch)
            *args: Positional arguments for func
            wait: If True, block until task completes and return result
            **kwargs: Keyword arguments for func
        
        Returns:
            If wait=True: Result from func
            If wait=False: None
        
        Raises:
            RuntimeError: If scheduler is shutdown
            Exception: If wait=True and func raises an exception
        """
        if self.shutdown_flag.is_set():
            raise RuntimeError("TrainingScheduler is shutdown")
        
        # Create result queue if caller wants to wait
        result_queue = queue.Queue(maxsize=1) if wait else None
        
        # Create and submit task
        task = TrainingTask(
            func=func,
            args=args,
            kwargs=kwargs,
            result_queue=result_queue
        )
        
        try:
            self.task_queue.put(task, timeout=5.0)
        except queue.Full:
            raise RuntimeError("Training queue is full - system is overloaded")
        
        # Wait for result if requested
        if wait:
            try:
                status, result = result_queue.get(timeout=30.0)
                if status == "success":
                    return result
                else:
                    raise RuntimeError(result)
            except queue.Empty:
                raise TimeoutError("Training task timed out after 30 seconds")
        
        return None
    
    def shutdown(self, timeout: float = 5.0):
        """
        Shutdown the training scheduler gracefully.
        
        Args:
            timeout: Maximum time to wait for pending tasks to complete
        """
        print("[TrainingScheduler] Shutting down...")
        self.shutdown_flag.set()
        
        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
        
        print("[TrainingScheduler] Shutdown complete")
    
    def wait_for_pending_tasks(self, timeout: float = 10.0):
        """
        Wait for all pending tasks in the queue to complete.
        
        Args:
            timeout: Maximum time to wait
        
        Returns:
            True if all tasks completed, False if timeout
        """
        try:
            # Queue.join() blocks until all tasks are done
            self.task_queue.join()
            return True
        except:
            return False


# Global singleton instance
_global_scheduler: Optional[TrainingScheduler] = None
_scheduler_lock = threading.Lock()


def get_training_scheduler() -> TrainingScheduler:
    """
    Get the global training scheduler singleton.
    Thread-safe lazy initialization.
    """
    global _global_scheduler
    
    if _global_scheduler is None:
        with _scheduler_lock:
            if _global_scheduler is None:
                _global_scheduler = TrainingScheduler()
                print("[TrainingScheduler] Initialized global training scheduler")
    
    return _global_scheduler


def shutdown_training_scheduler():
    """Shutdown the global training scheduler if it exists."""
    global _global_scheduler
    
    if _global_scheduler is not None:
        _global_scheduler.shutdown()
        _global_scheduler = None
