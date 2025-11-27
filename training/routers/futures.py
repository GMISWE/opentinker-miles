"""
Futures Router - Async Operation Result Retrieval

Thin HTTP layer for:
1. Retrieving async operation results
2. Cleaning up old futures
3. No business logic - just storage access
"""
import logging
import time
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any

from ..models.requests import CleanupFuturesRequest, RetrieveFutureRequest
from ..models.responses import CleanupResult
from ..storage import FuturesStorage
from ..core.dependencies import verify_api_key_dep

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    tags=["futures"]
)

def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Training runtime state not initialized")
    return runtime


def get_futures_storage(request: Request) -> FuturesStorage:
    """Get the futures_storage instance"""
    storage = getattr(request.app.state, "futures_storage", None)
    if storage is None:
        raise RuntimeError("Futures storage not initialized")
    return storage


def get_legacy_futures_store(request: Request) -> Dict:
    """Get the legacy futures_store"""
    runtime = _get_runtime(request)
    return runtime.futures_store


def get_poll_tracking(request: Request) -> Dict[str, Dict[str, Any]]:
    """Get the poll_tracking dict"""
    runtime = _get_runtime(request)
    return runtime.poll_tracking


@router.post("/api/v1/retrieve_future/{request_id}")
async def retrieve_future(
    request_id: str,
    _: None = Depends(verify_api_key_dep),
    futures_storage: FuturesStorage = Depends(get_futures_storage),
    futures_store: Dict = Depends(get_legacy_futures_store),
    poll_tracking: Dict[str, Dict[str, Any]] = Depends(get_poll_tracking)
):
    """
    Retrieve async operation result - refactored with storage abstraction

    Returns:
    - 408 (Request Timeout) if operation is still running
    - 200 with result if completed successfully
    - 500 if failed

    Benefits:
    - Storage abstraction instead of direct dict access
    - Proper HTTP status codes matching Tinker API spec
    - Cleaner error handling
    - Request ID in path for RESTful design
    """
    # Smart logging for polling operations
    if request_id not in poll_tracking:
        poll_tracking[request_id] = {
            "start_time": time.time(),
            "count": 0
        }
        logger.info(f"[retrieve_future] Started polling for {request_id}")

    poll_tracking[request_id]["count"] += 1
    poll_count = poll_tracking[request_id]["count"]

    # Log every 10th poll at INFO, others at DEBUG
    if poll_count % 10 == 0:
        logger.info(f"[retrieve_future] Still polling {request_id} (#{poll_count})")
    else:
        logger.debug(f"[retrieve_future] Poll #{poll_count} for {request_id}")

    # Get future from storage
    future = futures_storage.get_future(request_id)

    if not future:
        # Also check legacy store for backward compatibility
        if request_id in futures_store:
            future = futures_store[request_id]
        else:
            raise HTTPException(status_code=404, detail=f"Future {request_id} not found")

    # Return appropriate HTTP status code based on future status
    if future["status"] == "completed":
        # Log completion
        if request_id in poll_tracking:
            stats = poll_tracking.pop(request_id)
            duration = time.time() - stats["start_time"]
            logger.info(
                f"[retrieve_future] {request_id} completed: "
                f"{stats['count']} polls over {duration:.2f}s"
            )
        # Return 200 with result data only (matching original API)
        return JSONResponse(content=future.get("result", {}))

    elif future["status"] == "failed":
        # Log failure
        if request_id in poll_tracking:
            stats = poll_tracking.pop(request_id)
            duration = time.time() - stats["start_time"]
            logger.info(
                f"[retrieve_future] {request_id} failed: "
                f"{stats['count']} polls over {duration:.2f}s"
            )
        # Extract error message
        error = None
        result = future.get("result")
        if result and isinstance(result, dict) and "error" in result:
            error = result["error"]
        elif "error" in future:
            error = future["error"]
        # Return 500 with error
        raise HTTPException(status_code=500, detail=error or "Operation failed")

    else:  # status == "pending"
        # Return 408 (Request Timeout) while pending
        raise HTTPException(status_code=408, detail="Operation still in progress")


@router.post("/api/v1/retrieve_future")
async def retrieve_future_body(
    request: RetrieveFutureRequest,
    _: None = Depends(verify_api_key_dep),
    futures_storage: FuturesStorage = Depends(get_futures_storage),
    futures_store: Dict = Depends(get_legacy_futures_store),
    poll_tracking: Dict[str, Dict[str, Any]] = Depends(get_poll_tracking)
):
    """
    Retrieve future (body version for backward compatibility).

    Returns:
    - 408 (Request Timeout) if operation is still running
    - 200 with result if completed successfully
    - 500 if failed
    """
    # Delegate to the path-based version which implements the correct behavior
    return await retrieve_future(
        request.request_id,
        _,
        futures_storage,
        futures_store,
        poll_tracking
    )


@router.post("/api/v1/cleanup_futures", response_model=CleanupResult)
async def cleanup_futures(
    request: CleanupFuturesRequest,
    _: None = Depends(verify_api_key_dep),
    futures_storage: FuturesStorage = Depends(get_futures_storage),
    futures_store: Dict = Depends(get_legacy_futures_store)
):
    """
    Cleanup old futures - refactored with storage abstraction
    """
    try:
        # Clean up using storage
        removed_count = futures_storage.cleanup_old_futures(
            max_age_hours=request.max_age_hours
        )

        # Also clean legacy store
        cutoff_time = datetime.now().timestamp() - (request.max_age_hours * 3600)
        legacy_removed = 0

        for request_id in list(futures_store.keys()):
            future = futures_store[request_id]
            if future.get("created_at"):
                try:
                    created = datetime.fromisoformat(future["created_at"]).timestamp()
                    if created < cutoff_time:
                        del futures_store[request_id]
                        legacy_removed += 1
                except:
                    pass

        total_removed = removed_count + legacy_removed

        logger.info(f"Cleaned up {total_removed} old futures")

        return CleanupResult(
            futures_cleaned=total_removed,
            message=f"Successfully cleaned up {total_removed} futures older than {request.max_age_hours} hours"
        )

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
