"""
Task Manager - Eliminates Duplicated Async Pattern

This module provides a reusable pattern for managing background async tasks
with futures storage integration. Every endpoint in api_refactored.py follows
the same pattern:

1. Save future to storage
2. Create async task
3. Try/catch execution
4. Update future status on completion/failure

This manager encapsulates that pattern (DRY principle).
"""
import asyncio
import logging
from typing import Callable, Dict, Any, Optional, Awaitable
from ..storage import FuturesStorage

logger = logging.getLogger(__name__)


class TaskManager:
    """
    Manages background async tasks with automatic futures storage integration.

    Eliminates the duplicated try/except/update_status pattern found in every
    endpoint of api_refactored.py.

    Example usage:
        task_manager = TaskManager(futures_storage)

        async def my_business_logic():
            # Do work
            return {"result": "success"}

        request_id = task_manager.create_task(
            operation="train",
            model_id="model_123",
            payload=request.dict(),
            task_func=my_business_logic
        )
    """

    def __init__(self, futures_storage: FuturesStorage):
        """
        Initialize task manager.

        Args:
            futures_storage: Storage backend for task status tracking
        """
        self.futures_storage = futures_storage
        self._active_tasks: Dict[str, asyncio.Task] = {}

    def create_task(
        self,
        request_id: str,
        operation: str,
        model_id: str,
        payload: Dict[str, Any],
        task_func: Callable[[], Awaitable[Dict[str, Any]]]
    ) -> None:
        """
        Create and track a background async task with automatic error handling.

        This method:
        1. Saves future to storage with "pending" status
        2. Wraps task_func with error handling
        3. Updates storage on completion/failure
        4. Tracks task lifecycle

        Args:
            request_id: Unique identifier for this request
            operation: Operation name (e.g., "forward_backward", "optim_step")
            model_id: Associated model ID
            payload: Original request payload for debugging
            task_func: Async function that performs the actual work.
                      Must return Dict[str, Any] on success.

        Example:
            async def do_training():
                result = await service.train(data)
                return {"loss": result.loss, "grad_norm": result.grad_norm}

            task_manager.create_task(
                request_id="req_123",
                operation="train",
                model_id="model_abc",
                payload=request.dict(),
                task_func=do_training
            )
        """
        # Save future with pending status
        self.futures_storage.save_future(
            request_id=request_id,
            operation=operation,
            payload=payload,
            model_id=model_id
        )

        async def wrapped_task():
            """Wrapper that handles success/failure and storage updates"""
            try:
                logger.info(f"[{request_id}] Starting {operation} for {model_id}")

                # Execute the actual business logic
                result = await task_func()

                # Update storage with success
                self.futures_storage.update_status(
                    request_id=request_id,
                    status="completed",
                    result=result
                )

                logger.info(f"[{request_id}] {operation} completed successfully")

            except Exception as e:
                # Log full traceback for debugging
                import traceback
                logger.error(f"[{request_id}] {operation} failed: {e}")
                logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")

                # Update storage with error
                self.futures_storage.update_status(
                    request_id=request_id,
                    status="failed",
                    result={"error": str(e)}
                )

            finally:
                # Cleanup task reference
                if request_id in self._active_tasks:
                    del self._active_tasks[request_id]

        # Create and track the task
        task = asyncio.create_task(wrapped_task())
        self._active_tasks[request_id] = task

        # Prevent garbage collection
        task.add_done_callback(lambda t: None)

    def get_task(self, request_id: str) -> Optional[asyncio.Task]:
        """
        Get active task by request ID.

        Args:
            request_id: Request identifier

        Returns:
            Task if active, None if not found or completed
        """
        return self._active_tasks.get(request_id)

    def cancel_task(self, request_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            request_id: Request identifier

        Returns:
            True if task was cancelled, False if not found
        """
        task = self._active_tasks.get(request_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"[{request_id}] Task cancelled")
            return True
        return False

    async def wait_all(self, timeout: Optional[float] = None) -> None:
        """
        Wait for all active tasks to complete.

        Useful for graceful shutdown.

        Args:
            timeout: Maximum time to wait in seconds
        """
        if not self._active_tasks:
            return

        tasks = list(self._active_tasks.values())
        logger.info(f"Waiting for {len(tasks)} active tasks to complete...")

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for tasks, {len(self._active_tasks)} still active")

    @property
    def active_count(self) -> int:
        """Get count of currently active tasks"""
        return len(self._active_tasks)
