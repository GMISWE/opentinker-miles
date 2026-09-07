"""
Checkpoint Service - Business Logic for Model Checkpointing

Handles:
- Saving model weights to disk (delegates to backend)
- Saving weights for SGLang sampler
- Checkpoint metadata management
"""
import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from ..backends.base import TrainingBackend
from ..storage import MetadataStorage
from ..backends.checkpoint_interchange import resolve_checkpoint_root

logger = logging.getLogger(__name__)


class CheckpointService:
    """Service for managing model checkpoints and weights."""

    def __init__(self, backend: TrainingBackend):
        self.backend = backend

    @staticmethod
    def _next_step_id(client_info: Dict[str, Any]) -> int:
        """Monotonic per-model save counter (the backend's iteration label)."""
        client_info["save_counter"] = client_info.get("save_counter", 0) + 1
        return client_info["save_counter"]

    async def save_weights(
        self,
        model_id: str,
        request_id: str,
        path: str,
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage,
    ) -> Dict[str, Any]:
        """
        Save model weights to disk.

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        handle = client_info["backend_handle"]
        training_run_id = client_info["training_run_id"]

        # Generate checkpoint name and step_id
        checkpoint_name = path or f"checkpoint_{int(time.time())}"
        step_id = self._next_step_id(client_info)
        checkpoint_path = f"tinker://{training_run_id}/weights/{checkpoint_name}"

        logger.info("[%s] Saving weights for %s to %s", request_id, model_id, checkpoint_path)

        # Delegate actual save to backend
        await self.backend.save_checkpoint(handle, checkpoint_path, step_id=step_id)

        # Save checkpoint metadata
        metadata_storage.save_checkpoint(
            model_id=model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_data={
                "path": checkpoint_path,
                "created_at": datetime.now().isoformat(),
                "type": "manual_save",
            },
        )

        logger.info("[%s] Weights saved successfully", request_id)

        return {
            "path": checkpoint_path,
            "checkpoint_path": checkpoint_path,
            "step_id": step_id,
            "name": checkpoint_name,
            "type": "save_weights",
        }

    async def load_weights(
        self,
        model_id: str,
        request_id: str,
        path: str,
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage,
        optimizer: bool = False,
    ) -> Dict[str, Any]:
        """Load a training checkpoint into a live model: weights, plus optimizer
        state when asked (the backend refuses rather than partially resumes)."""
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")
        handle = training_clients[model_id]["backend_handle"]
        logger.info("[%s] Loading %s for %s from %s", request_id,
                    "weights + optimizer state" if optimizer else "weights", model_id, path)
        await self.backend.load_checkpoint(handle, path, optimizer=optimizer)
        metadata_storage.update_training_run(
            training_clients[model_id].get("training_run_id", model_id),
            {"loaded_from": path, "last_request_time": datetime.now().isoformat()},
        )
        return {"type": "load_weights", "path": path, "model_id": model_id}

    @staticmethod
    def delete_checkpoint(
        model_id: str,
        checkpoint_type: str,
        checkpoint_id: str,
        metadata_storage: MetadataStorage,
    ) -> bool:
        """Remove a checkpoint's metadata record and its bytes. Returns False if unknown."""
        kind = "weights" if checkpoint_type == "training" else "sampler_weights"
        meta_name = checkpoint_id if kind == "weights" else f"sampler_{checkpoint_id}"
        if metadata_storage.load_checkpoint(model_id, meta_name) is None:
            return False
        root = resolve_checkpoint_root(f"tinker://{model_id}/{kind}/{checkpoint_id}")
        if os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)
        metadata_storage.delete_checkpoint(model_id, meta_name)
        logger.info("Deleted %s checkpoint %s of %s (%s)", checkpoint_type, checkpoint_id, model_id, root)
        return True

    @staticmethod
    def list_checkpoints(model_id: str, metadata_storage: MetadataStorage) -> list:
        """Tinker-shaped checkpoint records for a training run."""
        out = []
        for rec in metadata_storage.list_checkpoints(model_id, limit=1000):
            is_sampler = rec.get("type") == "sampler"
            path = rec.get("tinker_uri") if is_sampler else rec.get("path")
            if not path:
                continue
            name = path.removeprefix("tinker://").split("/")[-1]
            kind = "sampler_weights" if is_sampler else "weights"
            root = resolve_checkpoint_root(path)
            size = None
            if os.path.isdir(root):
                size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs)
            out.append({
                "checkpoint_id": f"{kind}/{name}",
                "checkpoint_type": "sampler" if is_sampler else "training",
                "time": rec.get("created_at"),
                "tinker_path": path,
                "size_bytes": size,
            })
        return out

    async def save_weights_for_sampler(
        self,
        model_id: str,
        request_id: str,
        name: Optional[str],
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage,
        path: Optional[str] = None,
        sampling_session_seq_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Save weights for SGLang sampler.

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        handle = client_info["backend_handle"]
        training_run_id = client_info.get("training_run_id", model_id)

        # Check if this is an ephemeral save (sampling_session_seq_id provided, no path/name)
        is_ephemeral = sampling_session_seq_id is not None and path is None and name is None
        logger.info(
            "[%s] save_weights_for_sampler: sampling_session_seq_id=%s, path=%r, name=%r, is_ephemeral=%s",
            request_id, sampling_session_seq_id, path, name, is_ephemeral,
        )

        if is_ephemeral:
            # Ephemeral save — generate sampling_session_id, don't persist path.
            # Skip save_model: weights are already synced to SGLang via update_weights().
            sampling_session_id = f"{model_id}_{sampling_session_seq_id}_{uuid.uuid4().hex[:8]}"
            logger.info("[%s] Ephemeral save for sampler: %s -> %s", request_id, model_id, sampling_session_id)
            logger.info("[%s] Skipping save_model for ephemeral save (weights already in SGLang)", request_id)

            return {
                "path": None,
                "sampling_session_id": sampling_session_id,
                "type": "save_weights_for_sampler",
            }
        else:
            # Persistent save — use path/name
            logger.info("[%s] Saving weights for sampler: %s", request_id, model_id)

            checkpoint_name = path or name or f"sampler_{int(time.time())}"
            step_id = self._next_step_id(client_info)
            tinker_uri = f"tinker://{training_run_id}/sampler_weights/{checkpoint_name}"
            # The backend resolves the URI through resolve_checkpoint_root, so the
            # recorded filesystem path is where the bytes actually land.
            checkpoint_path = resolve_checkpoint_root(tinker_uri)

            # Delegate actual save to backend
            await self.backend.save_checkpoint(handle, tinker_uri, step_id=step_id)

            # Save checkpoint metadata
            metadata_storage.save_checkpoint(
                model_id=model_id,
                checkpoint_name=f"sampler_{checkpoint_name}",
                checkpoint_data={
                    "path": checkpoint_path,
                    "tinker_uri": tinker_uri,
                    "created_at": datetime.now().isoformat(),
                    "type": "sampler",
                    "step_id": step_id,
                },
            )

            logger.info("[%s] Weights saved to %s", request_id, tinker_uri)

            return {
                "path": tinker_uri,
                "sampling_session_id": None,
                "checkpoint_path": checkpoint_path,
                "step_id": step_id,
                "name": checkpoint_name,
                "type": "save_weights_for_sampler",
            }
