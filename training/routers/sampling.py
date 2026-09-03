"""
Sampling Router - HTTP Layer for Model Sampling

Endpoints:
- POST /api/v1/asample - Async sampling via SGLang
- POST /api/v1/sample - Sync sampling via SGLang
- POST /api/v1/create_sampling_client - Create SGLang sampling client
"""
import logging
from typing import Dict, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..services.sampling_service import SamplingService
from ..core.task_manager import TaskManager
from ..core.dependencies import verify_api_key_dep
from ..storage import FuturesStorage
from ..models.requests import (
    ASampleRequest,
    SampleRequest,
    CreateSamplingClientRequest,
)
from ..models.responses import (
    AsyncOperationResponse,
    SampleResult,
    SamplingSequence,
    CreateSamplingClientResult,
)
from ..utils import generate_request_id

logger = logging.getLogger(__name__)

router = APIRouter()

def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Training runtime state not initialized")
    return runtime


def get_sampling_service(request: Request) -> SamplingService:
    """Dependency injection for SamplingService."""
    service = getattr(request.app.state, "sampling_service", None)
    if service is None:
        raise RuntimeError("SamplingService not initialized on app state")
    return service


def get_futures_storage(request: Request) -> FuturesStorage:
    """Dependency injection for FuturesStorage."""
    storage = getattr(request.app.state, "futures_storage", None)
    if storage is None:
        raise RuntimeError("FuturesStorage not initialized on app state")
    return storage


def get_training_clients(request: Request) -> Dict[str, Dict[str, Any]]:
    """Dependency injection for training_clients."""
    runtime = _get_runtime(request)
    return runtime.training_clients


def get_task_manager(
    futures_storage: FuturesStorage = Depends(get_futures_storage)
) -> TaskManager:
    """Create TaskManager with FuturesStorage dependency."""
    return TaskManager(futures_storage)


# ============================================================================
# Sampling Endpoints
# ============================================================================

def get_session_service(request: Request):
    """Session service from app state (None if unavailable)."""
    return getattr(request.app.state, "session_service", None)


BASE_MODEL_SAMPLING_UNSUPPORTED = (
    "base-model sampling is not supported by this deployment; create a model "
    "and a sampler from it (save_weights_and_get_sampling_client) instead"
)


def resolve_target_model(
    training_clients: Dict,
    session_service,
    sampling_session_id: Optional[str] = None,
    model_path: Optional[str] = None,
    base_model: Optional[str] = None,
) -> tuple:
    """Return (model_id, pinned_version) for a sampling request, or raise.

    A sampler names its owning model (registered at save_weights_for_sampler /
    create_sampling_session); a bare model_path names it by URI. There is no
    fallback to "some model": under a multi-tenant pool that serves a
    co-tenant's adapter, and a bare base_model has no engine behind it.
    """
    if sampling_session_id:
        info = session_service.get_sampler(sampling_session_id) if session_service is not None else None
        if info is None:
            raise HTTPException(status_code=404, detail=f"Unknown sampling_session_id: {sampling_session_id}")
        if not info.model_id:
            raise HTTPException(status_code=400, detail=BASE_MODEL_SAMPLING_UNSUPPORTED)
        if info.model_id not in training_clients:
            raise HTTPException(status_code=404, detail=f"Sampler {sampling_session_id}'s model {info.model_id} no longer exists")
        return info.model_id, getattr(info, "pinned_version", None)
    if model_path:
        model_id = model_id_from_path(model_path)
        if model_id not in training_clients:
            raise HTTPException(status_code=404, detail=f"Model {model_id!r} from model_path {model_path!r} not found")
        return model_id, None
    if base_model:
        raise HTTPException(status_code=400, detail=BASE_MODEL_SAMPLING_UNSUPPORTED)
    raise HTTPException(status_code=400, detail="Provide sampling_session_id or a tinker:// model_path")


def model_id_from_path(model_path: str) -> str:
    """tinker://<model_id>/... -> model_id."""
    return model_path.removeprefix("tinker://").split("/")[0]


@router.post("/api/v1/asample", response_model=AsyncOperationResponse)
async def asample(
    request: ASampleRequest,
    _: None = Depends(verify_api_key_dep),
    service: SamplingService = Depends(get_sampling_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients),
    session_service=Depends(get_session_service),
):
    """
    Async sampling via SGLang.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    # Extract prompt tokens
    prompt_tokens = request.prompt.get_tokens()

    # BUG-015: resolve the sampler's pinned weight version (snapshot samplers,
    # e.g. DPO's frozen reference) so pinned logprob reads aren't served from
    # the live refit-every-step engine. The sampler also names its OWNING
    # model — required routing under a multi-tenant pool, where find-first
    # would serve a co-tenant's adapter.
    model_id, pinned_version = resolve_target_model(
        training_clients, session_service,
        sampling_session_id=request.sampling_session_id,
        model_path=request.model_path, base_model=request.base_model,
    )
    target_model_id = model_id

    async def execute():
        result_dict = await service.async_sample(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            num_samples=request.num_samples,
            sampling_params=request.sampling_params.dict() if request.sampling_params else None,
            prompt_logprobs=request.prompt_logprobs,
            training_clients=training_clients,
            pinned_version=pinned_version,
            model_id=target_model_id,
        )

        # Convert to response model
        sequences = [SamplingSequence(**seq) for seq in result_dict["sequences"]]
        return SampleResult(
            sequences=sequences,
            prompt_logprobs=result_dict.get("prompt_logprobs"),
            weight_version=result_dict.get("weight_version"),
            latest_weight_version=result_dict.get("latest_weight_version"),
        )

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="asample",
        model_id=model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=model_id
    )


@router.post("/api/v1/sample", response_model=AsyncOperationResponse)
async def sample(
    request: SampleRequest,
    _: None = Depends(verify_api_key_dep),
    service: SamplingService = Depends(get_sampling_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients),
    session_service=Depends(get_session_service),
):
    """
    Synchronous sampling via SGLang.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    model_id, _ = resolve_target_model(
        training_clients, session_service,
        sampling_session_id=request.sampling_session_id,
        model_path=request.model_path, base_model=request.base_model,
    )

    async def execute():
        result_dict = await service.sync_sample(
            request_id=request_id,
            prompts=request.prompts,
            num_samples=request.num_samples,
            sampling_params=request.sampling_params.dict() if request.sampling_params else None,
            training_clients=training_clients,
            model_id=model_id,
        )

        # Convert to response model
        sequences = [SamplingSequence(**seq) for seq in result_dict["sequences"]]
        return SampleResult(sequences=sequences)

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="sample",
        model_id=model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=model_id
    )


@router.post("/api/v1/create_sampling_client", response_model=AsyncOperationResponse)
async def create_sampling_client(
    request: CreateSamplingClientRequest,
    _: None = Depends(verify_api_key_dep),
    service: SamplingService = Depends(get_sampling_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients),
    session_service=Depends(get_session_service),
):
    """
    Create sampling client bound to the model named by model_path.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    model_id, _ = resolve_target_model(
        training_clients, session_service,
        model_path=request.model_path, base_model=request.base_model,
    )

    async def execute():
        result_dict = await service.create_sampling_client(
            request_id=request_id,
            model_path=request.model_path,
            training_clients=training_clients,
            model_id=model_id,
        )
        return CreateSamplingClientResult(**result_dict)

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="create_sampling_client",
        model_id=model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=model_id
    )
