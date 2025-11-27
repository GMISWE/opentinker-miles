"""
Checkpoint Service - Business Logic for Model Checkpointing

Handles:
- Saving model weights to disk
- Saving weights for SGLang sampler
- Checkpoint metadata management
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any

from ..storage import MetadataStorage
from ..utils.helpers import generate_step_id

logger = logging.getLogger(__name__)


class CheckpointService:
    """Service for managing model checkpoints and weights."""

    def __init__(self):
        """Initialize CheckpointService."""
        pass

    async def save_weights(
        self,
        model_id: str,
        request_id: str,
        path: str,
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage
    ) -> Dict[str, Any]:
        """
        Save model weights to disk.

        Args:
            model_id: Model identifier
            request_id: Request identifier for logging
            path: Optional checkpoint name/path
            training_clients: Global training clients dict
            metadata_storage: Metadata storage instance

        Returns:
            Dict with checkpoint_path

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        train_group = client_info["train_group"]
        training_run_id = client_info["training_run_id"]

        # Generate checkpoint name and step_id
        checkpoint_name = path or f"checkpoint_{int(time.time())}"
        step_id = generate_step_id(checkpoint_name)
        checkpoint_path = f"tinker://{training_run_id}/weights/{checkpoint_name}"

        logger.info(f"[{request_id}] Saving weights for {model_id} to {checkpoint_path}")

        # Save weights using async Ray API (matching original api.py pattern)
        # Call save_model.remote() on each actor directly
        object_refs = [
            actor.save_model.remote(step_id)
            for actor in train_group._actor_handlers
        ]

        # Await all save operations
        await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in object_refs])

        # Save checkpoint metadata
        metadata_storage.save_checkpoint(
            model_id=model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_data={
                "path": checkpoint_path,
                "created_at": datetime.now().isoformat(),
                "type": "manual_save"
            }
        )

        logger.info(f"[{request_id}] Weights saved successfully")

        # Return format matching original API (for backward compatibility)
        return {
            "path": checkpoint_path,  # Tinker URI format
            "checkpoint_path": checkpoint_path,  # Keep for internal use
            "step_id": step_id,
            "name": checkpoint_name,
            "type": "save_weights"
        }

    async def save_weights_for_sampler(
        self,
        model_id: str,
        request_id: str,
        name: str,
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage
    ) -> Dict[str, Any]:
        """
        Save weights for SGLang sampler.

        Args:
            model_id: Model identifier
            request_id: Request identifier for logging
            name: Optional checkpoint name
            training_clients: Global training clients dict
            metadata_storage: Metadata storage instance

        Returns:
            Dict with path, checkpoint_path, step_id, name

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        train_group = client_info["train_group"]
        training_run_id = client_info.get("training_run_id", model_id)

        logger.info(f"[{request_id}] Saving weights for sampler: {model_id}")

        # Generate checkpoint name and path
        checkpoint_name = name or f"sampler_{int(time.time())}"
        step_id = generate_step_id(checkpoint_name)
        checkpoint_path = f"/data/checkpoints/tinker/iter_{step_id:07d}"
        tinker_uri = f"tinker://{training_run_id}/weights/{checkpoint_name}"

        # Save weights using async Ray API (matching original api.py pattern)
        object_refs = [
            actor.save_model.remote(step_id)
            for actor in train_group._actor_handlers
        ]

        # Await all save operations
        await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in object_refs])

        # Save checkpoint metadata
        metadata_storage.save_checkpoint(
            model_id=model_id,
            checkpoint_name=f"sampler_{checkpoint_name}",
            checkpoint_data={
                "path": checkpoint_path,
                "tinker_uri": tinker_uri,
                "created_at": datetime.now().isoformat(),
                "type": "sampler",
                "step_id": step_id
            }
        )

        logger.info(f"[{request_id}] Weights saved to {tinker_uri}")

        return {
            "path": tinker_uri,
            "checkpoint_path": checkpoint_path,
            "step_id": step_id,
            "name": checkpoint_name
        }
