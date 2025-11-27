"""
SGLang Inference Client

Async HTTP client for SGLang inference server.
Handles generation with defensive prompt logprobs extraction.
"""
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class SGLangClient:
    """
    Async HTTP client for SGLang inference server.

    Handles text generation with support for:
    - Input token IDs
    - Sampling parameters (temperature, top_p, max_tokens)
    - Prompt logprobs extraction with defensive None handling
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        """
        Initialize SGLang client.

        Args:
            base_url: SGLang server URL (e.g., "http://router:8000")
            timeout: HTTP request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.generate_url = f"{self.base_url}/generate"

    async def generate(
        self,
        input_ids: List[int],
        sampling_params: Optional[Dict[str, Any]] = None,
        prompt_logprobs: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text completion from SGLang.

        Args:
            input_ids: List of input token IDs
            sampling_params: Sampling parameters dict with:
                - temperature: float (default 0.7)
                - top_p: float (default 0.9)
                - max_tokens: int (default 256)
            prompt_logprobs: Whether to return prompt log probabilities

        Returns:
            Dict with:
                - tokens: List[int] - Generated token IDs
                - logprobs: List[float] - Log probabilities for generated tokens
                - text: Optional[str] - Generated text
                - stop_reason: str - "stop" or "length"
                - prompt_logprobs: Optional[List[Optional[float]]] - Prompt logprobs if requested

        Raises:
            httpx.HTTPError: If SGLang request fails
            ValueError: If response format is invalid
        """
        # Build request payload
        sampling_params = sampling_params or {}
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": sampling_params.get("temperature", 0.7),
                "top_p": sampling_params.get("top_p", 0.9),
                "max_new_tokens": sampling_params.get("max_tokens", 256),
            },
            "return_logprob": True,
        }

        # Request prompt logprobs if needed
        if prompt_logprobs:
            payload["logprob_start_len"] = 0
            logger.debug(f"Requesting prompt logprobs from SGLang")

        # Make async HTTP request
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.generate_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                sglang_output = response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"SGLang HTTP error {e.response.status_code}: {e.response.text}")
                raise
            except httpx.RequestError as e:
                logger.error(f"SGLang request error: {e}")
                raise
            except Exception as e:
                logger.error(f"SGLang unexpected error: {e}")
                raise

        # Extract tokens and logprobs from response
        try:
            meta_info = sglang_output["meta_info"]

            # Output tokens and logprobs
            # Format: [(logprob, token_id), ...]
            token_logprobs = meta_info["output_token_logprobs"]
            output_tokens = [item[1] for item in token_logprobs]
            output_logprobs = [item[0] for item in token_logprobs]

            # Build result
            result = {
                "tokens": output_tokens,
                "logprobs": output_logprobs,
                "text": sglang_output.get("text"),
                "stop_reason": "stop" if sglang_output.get("finish_reason") == "stop" else "length"
            }

            # Extract prompt logprobs if requested
            if prompt_logprobs:
                input_logprobs = meta_info.get("input_token_logprobs", [])

                # Defensive handling: SGLang's input_token_logprobs format was undocumented
                # and had a bug where it already includes None prefix.
                # See CLAUDE.md -> "Logprobs None Handling" section.
                #
                # This code works regardless of SGLang format:
                # - If SGLang returns [None, logprob1, ...] -> use as-is
                # - If SGLang returns [logprob1, ...] -> prepend None
                if input_logprobs and input_logprobs[0] is None:
                    result["prompt_logprobs"] = input_logprobs
                    logger.debug(f"SGLang already has None prefix in input_token_logprobs")
                else:
                    result["prompt_logprobs"] = [None] + input_logprobs
                    logger.debug(f"Prepended None to input_token_logprobs")

            logger.debug(f"Generated {len(output_tokens)} tokens from SGLang")
            return result

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse SGLang response: {e}")
            logger.error(f"SGLang response: {sglang_output}")
            raise ValueError(f"Invalid SGLang response format: {e}")

    async def batch_generate(
        self,
        input_ids_list: List[List[int]],
        sampling_params: Optional[Dict[str, Any]] = None,
        prompt_logprobs: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple completions (one per input).

        Args:
            input_ids_list: List of input token ID lists
            sampling_params: Sampling parameters
            prompt_logprobs: Whether to return prompt logprobs

        Returns:
            List of generation results (one per input)
        """
        results = []
        for input_ids in input_ids_list:
            result = await self.generate(input_ids, sampling_params, prompt_logprobs)
            results.append(result)
        return results
