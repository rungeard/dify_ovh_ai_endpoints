from typing import Mapping, Optional
import json
import re
import logging

import httpx
import tiktoken

from dify_plugin.entities.model import (
    AIModelEntity,
    EmbeddingInputType,
    I18nObject,
    ModelFeature,
)
from dify_plugin.entities.model.text_embedding import (
    TextEmbeddingResult,
    EmbeddingUsage,
)
from dify_plugin.errors.model import InvokeError, InvokeServerUnavailableError
from dify_plugin.interfaces.model.openai_compatible.text_embedding import (
    OAICompatEmbeddingModel,
)
from models.ovh_errors import format_ovh_rate_limit_error
from models.ovh_credentials import build_ovh_credentials


logger = logging.getLogger(__name__)


class OpenAITextEmbeddingModel(OAICompatEmbeddingModel):
    def get_customizable_model_schema(
        self, model: str, credentials: Mapping | dict
    ) -> AIModelEntity:
        credentials = credentials or {}
        entity = super().get_customizable_model_schema(model, credentials)

        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label = I18nObject(en_US=credentials["display_name"])

        # Add vision feature if vision support is enabled
        vision_support = credentials.get("vision_support", "no_support")
        if vision_support == "support":
            if entity.features is None:
                entity.features = []
            if ModelFeature.VISION not in entity.features:
                entity.features.append(ModelFeature.VISION)

        return entity

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        Invoke text embedding model with multimodal support

        Supports text inputs and degrades image references to text markers when
        vision-style input is provided.

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed (can be JSON strings for multimodal)
        :param user: unique user id
        :param input_type: input type
        :return: embeddings result
        """
        credentials = build_ovh_credentials(credentials)
        # Check if vision support is enabled
        vision_support = credentials.get("vision_support", "no_support")

        # Process inputs - convert to multimodal format if needed
        processed_inputs = [
            self._process_input(text, vision_support == "support")
            for text in texts
        ]

        # Apply prefix
        prefix = self._get_prefix(credentials, input_type)
        if prefix:
            processed_inputs = self._add_prefix_to_inputs(processed_inputs, prefix)

        # Get context size and max chunks from credentials or model properties
        context_size = self._get_context_size(model, credentials)
        max_chunks = self._get_max_chunks(model, credentials)

        # Truncate long texts to fit the embedding context window.
        inputs = []
        for text in processed_inputs:
            # Check token count and truncate if necessary
            num_tokens = self._get_num_tokens_by_gpt2(text)
            if num_tokens >= context_size:
                # Truncate to fit within context size
                cutoff = int(len(text) * (context_size / num_tokens))
                text = text[0:cutoff]

            inputs.append(text)

        # Call API in batches
        return self._embed_in_batches(model, credentials, inputs, user, input_type)

    def _embed_in_batches(
        self,
        model: str,
        credentials: dict,
        inputs: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        Embed texts in batches, handling API limits.
        """
        endpoint_url = credentials.get("endpoint_url", "").rstrip("/")
        api_key = credentials.get("api_key", "")
        endpoint_model_name = credentials.get("endpoint_model_name", "") or model
        max_chunks = self._get_max_chunks(model, credentials)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        }

        batched_embeddings = []
        used_tokens = 0
        total_price = 0.0

        # Initialize with default values, will be updated from API response if available
        unit_price = 0.0
        price_unit = 0.0
        currency = "USD"

        try:
            # Process in batches
            for i in range(0, len(inputs), max_chunks):
                batch = inputs[i : i + max_chunks]

                payload = {
                    "model": endpoint_model_name,
                    "input": batch,
                }

                # Add encoding_format only if specified in credentials
                encoding_format = credentials.get("encoding_format")
                if encoding_format:
                    payload["encoding_format"] = encoding_format

                response = httpx.post(
                    f"{endpoint_url}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                # Debug: log response on error
                if response.status_code != 200:
                    error_body = format_ovh_rate_limit_error(response.status_code, response.text, api_key)
                    logger.error(
                        f"Embedding API Error {response.status_code}: {error_body}"
                    )
                    raise InvokeError(error_body)

                response.raise_for_status()

                result = response.json()

                # Extract embeddings
                for data in result["data"]:
                    batched_embeddings.append(data["embedding"])

                # Extract usage information from API response
                usage = result.get("usage") or {}
                tokens = usage.get("prompt_tokens") or usage.get("total_tokens") or 0
                used_tokens += tokens

                # Extract pricing information if provided by API
                total_price += usage.get("total_price", 0.0)

                # Use API provided values if available, otherwise keep defaults
                if "unit_price" in usage:
                    unit_price = usage.get("unit_price", 0.0)
                if "price_unit" in usage:
                    price_unit = usage.get("price_unit", 0.0)
                if "currency" in usage:
                    currency = usage.get("currency", "USD")

            return TextEmbeddingResult(
                embeddings=batched_embeddings,
                model=model,
                usage=EmbeddingUsage(
                    tokens=used_tokens,
                    total_tokens=used_tokens,
                    unit_price=unit_price,
                    price_unit=price_unit,
                    total_price=total_price,
                    currency=currency,
                    latency=0.0,  # Latency tracking would require timing each request
                ),
            )

        except httpx.HTTPError as ex:
            raise InvokeServerUnavailableError(str(ex))
        except Exception as ex:
            raise InvokeError(str(ex))

    def _process_input(self, text: str, vision_enabled: bool) -> str:
        """
        Normalize input to plain text before calling the embedding endpoint.

        The OVH embedding endpoint used by this provider is text-oriented.
        When Dify sends vision-style content, image references are preserved as
        textual markers instead of being forwarded as multimodal payloads.
        """
        if not vision_enabled:
            return text

        # Try to parse as JSON
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return self._format_multimodal_content(data)
        except json.JSONDecodeError:
            pass

        # Try to detect markdown image syntax: ![desc](url)
        content = self._extract_markdown_images(text)
        if content != text:
            return content

        # Try to detect plain image URLs
        if self._is_image_url(text):
            return self._format_image_reference(text)

        return text

    def _format_multimodal_content(self, data: dict) -> str:
        """
        Format a simple multimodal-like dict into plain text.

        Expected format: {"text": "...", "image": "url_or_path"}
        """
        parts: list[str] = []
        if data.get("text"):
            parts.append(str(data["text"]))
        if data.get("image"):
            parts.append(self._format_image_reference(str(data["image"])))
        return " ".join(part for part in parts if part).strip()

    def _extract_markdown_images(self, text: str) -> str:
        """
        Replace markdown image syntax with a plain text marker.
        """
        pattern = r"!\[([^\]]*)\]\(([^\)]+)\)"
        return re.sub(pattern, lambda match: self._format_image_reference(match.group(2)), text)

    def _is_image_url(self, text: str) -> bool:
        """Check if text is an image URL."""
        image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg")
        return text.startswith(("http://", "https://")) and any(
            text.lower().endswith(ext) for ext in image_extensions
        )

    @staticmethod
    def _format_image_reference(image: str) -> str:
        image = image.strip()
        if image.startswith("data:image"):
            return "[Image: embedded data URI]"
        return f"[Image: {image}]"

    @staticmethod
    def _add_prefix_to_inputs(inputs: list[str], prefix: str) -> list[str]:
        """Add a textual prefix to each embedding input."""
        return [f"{prefix} {item}" for item in inputs]

    def _get_prefix(self, credentials: dict, input_type: EmbeddingInputType) -> str:
        if input_type == EmbeddingInputType.DOCUMENT:
            return credentials.get("document_prefix", "")

        if input_type == EmbeddingInputType.QUERY:
            return credentials.get("query_prefix", "")

        return ""

    def _get_num_tokens_by_gpt2(self, text: str) -> int:
        """
        Get token count for text using GPT-2 tokenizer (tiktoken).

        :param text: text to count tokens for
        :return: number of tokens (approximate)
        """
        try:
            encoding = tiktoken.get_encoding("gpt2")
            return len(encoding.encode(text))
        except Exception:
            # Fallback to character count or a default if tiktoken fails
            return (
                len(text) // 4
            )  # Rough estimate if tiktoken is not available or fails
