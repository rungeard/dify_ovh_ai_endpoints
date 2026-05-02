import json
import re
from contextlib import suppress
from typing import Mapping, Optional, Union, Generator, List
from urllib.parse import urljoin

import httpx
from dify_plugin.entities.model import (
    AIModelEntity,
    DefaultParameterName,
    I18nObject,
    ModelFeature,
    ParameterRule,
    ParameterType,
)
from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool,
    AssistantPromptMessage,
)
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin.interfaces.model.openai_compatible.llm import OAICompatLargeLanguageModel

from models.ovh_credentials import build_ovh_credentials


class OpenAILargeLanguageModel(OAICompatLargeLanguageModel):
    # Pre-compiled regex for better performance
    _THINK_PATTERN = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)
    # Models that require max_completion_tokens (OpenAI Responses API family)
    _NEEDS_MAX_COMPLETION_TOKENS_PATTERN = re.compile(r"^(o1|o3|gpt-5)", re.IGNORECASE)
    _OVH_ALLOWED_CHAT_PARAMETERS = {
        "frequency_penalty",
        "json_schema",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
    }
    _OVH_ALLOWED_COMPLETION_PARAMETERS = {
        "best_of",
        "echo",
        "frequency_penalty",
        "json_schema",
        "logprobs",
        "max_tokens",
        "n",
        "presence_penalty",
        "seed",
        "stop",
        "suffix",
        "temperature",
        "top_p",
    }

    def _wrap_thinking_by_reasoning_content(self, delta: dict, is_reasoning: bool) -> tuple[str, bool]:
        """
        Override base wrapper to support OVH `reasoning` deltas, emitting
        <think> blocks compatible with Dify's downstream filters.
        """
        reasoning_piece = delta.get("reasoning") or ""
        content_piece = delta.get("content") or ""

        if reasoning_piece:
            if not is_reasoning:
                # Open a think block on first reasoning token
                output = f"<think>\n{reasoning_piece}"
                is_reasoning = True
            else:
                # Continue streaming inside the think block
                output = str(reasoning_piece)
        elif is_reasoning:
            # No reasoning token in this delta, close the think block
            is_reasoning = False
            output = f"\n</think>{content_piece}"
        else:
            # No reasoning token and not in a reasoning block
            output = content_piece

        return output, is_reasoning

    # Timeout for validation requests: (connect_timeout, read_timeout) in seconds
    _VALIDATE_TIMEOUT = (10, 300)

    @staticmethod
    def _needs_max_completion_tokens(m: str) -> bool:
        return bool(OpenAILargeLanguageModel._NEEDS_MAX_COMPLETION_TOKENS_PATTERN.match(m))

    @staticmethod
    def _raise_credentials_error(response: httpx.Response) -> None:
        """Raise a CredentialsValidateFailedError with response details."""
        raise CredentialsValidateFailedError(
            f"Credentials validation failed with status code {response.status_code} "
            f"and response body {response.text}"
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """Validate credentials with fallback handling for multiple error scenarios.

        1) Try base validation first (keeps upstream compatibility).
        2) If it fails due to too-small token floor on Responses API
           (e.g., "Invalid 'max_output_tokens' ... integer_below_min_value"),
           retry once with a safe minimum of 16 using the appropriate endpoint/param.
        """
        credentials = build_ovh_credentials(credentials)
        # When max_completion_tokens is explicitly requested, validate directly
        # instead of letting the base class fail with max_tokens first.
        param_pref = credentials.get("token_param_name", "auto")
        endpoint_model = credentials.get("endpoint_model_name") or model
        if (
            param_pref == "max_completion_tokens"
            or (param_pref == "auto" and self._needs_max_completion_tokens(endpoint_model))
        ):
            self._retry_with_safe_min_tokens(model, credentials)
            return

        try:
            return super().validate_credentials(model, credentials)
        except CredentialsValidateFailedError as e:
            msg = str(e)

            # --- Retry path 1: token parameter incompatibility ---
            should_retry_floor = (
                "Invalid 'max_output_tokens'" in msg
                or "integer_below_min_value" in msg
            )
            if should_retry_floor:
                self._retry_with_safe_min_tokens(model, credentials)
                return

            # Propagate unrelated validation errors
            raise

    def _retry_with_safe_min_tokens(self, model: str, credentials: dict) -> None:
        """Retry validation with a safe minimum token count for Responses API."""
        endpoint_url = credentials.get("endpoint_url")
        if not endpoint_url:
            raise CredentialsValidateFailedError("Missing endpoint_url in credentials")

        api_key = credentials.get("api_key")
        endpoint_model = credentials.get("endpoint_model_name") or model
        mode = credentials.get("mode", "chat")

        param_pref = credentials.get("token_param_name", "auto")
        use_max_completion = (
            param_pref == "max_completion_tokens"
            or (param_pref == "auto" and self._needs_max_completion_tokens(endpoint_model))
        )

        SAFE_MIN_TOKENS = 16

        try:
            headers = {"Content-Type": "application/json"}
            headers["Authorization"] = f"Bearer {api_key}"

            if mode == "chat":
                endpoint = urljoin(endpoint_url.rstrip("/") + "/", "chat/completions")
                payload = {
                    "model": endpoint_model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                }
                if use_max_completion:
                    payload["max_completion_tokens"] = SAFE_MIN_TOKENS
                else:
                    payload["max_tokens"] = SAFE_MIN_TOKENS
            else:
                endpoint = urljoin(endpoint_url.rstrip("/") + "/", "completions")
                payload = {
                    "model": endpoint_model,
                    "prompt": "ping",
                    "max_tokens": SAFE_MIN_TOKENS,
                    "stream": False,
                }

            response = httpx.post(endpoint, headers=headers, json=payload, timeout=self._VALIDATE_TIMEOUT)
            if response.status_code != 200:
                self._raise_credentials_error(response)
        except Exception as sub_e:
            raise CredentialsValidateFailedError(str(sub_e)) from sub_e

    def get_customizable_model_schema(
        self, model: str, credentials: Mapping | dict
    ) -> AIModelEntity:
        credentials = build_ovh_credentials(credentials)
        entity = super().get_customizable_model_schema(model, credentials)

        structured_output_support = credentials.get("structured_output_support", "not_supported")
        if structured_output_support == "supported":
            # ----
            # The following section should be added after the new version of `dify-plugin-sdks`
            # is released.
            # Related Commit:
            # https://github.com/langgenius/dify-plugin-sdks/commit/0690573a879caf43f92494bf411f45a1835d96f6
            # ----
            # try:
            #     entity.features.index(ModelFeature.STRUCTURED_OUTPUT)
            # except ValueError:
            #     entity.features.append(ModelFeature.STRUCTURED_OUTPUT)

            entity.parameter_rules.append(
                ParameterRule(
                    name=DefaultParameterName.RESPONSE_FORMAT.value,
                    label=I18nObject(en_US="Response Format"),
                    help=I18nObject(
                        en_US="Specifying the format that the model must output.",
                    ),
                    type=ParameterType.STRING,
                    options=["text", "json_object", "json_schema"],
                    required=False,
                )
            )
            entity.parameter_rules.append(
                ParameterRule(
                    name="reasoning_format",
                    label=I18nObject(en_US="Reasoning Format"),
                    help=I18nObject(
                        en_US="Specifying the format that the model must output reasoning.",
                    ),
                    type=ParameterType.STRING,
                    options=["none", "auto", "deepseek", "deepseek-legacy"],
                    required=False,
                )
            )
            entity.parameter_rules.append(
                ParameterRule(
                    name=DefaultParameterName.JSON_SCHEMA.value,
                    use_template=DefaultParameterName.JSON_SCHEMA.value,
                )
            )

        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label = I18nObject(en_US=credentials["display_name"])
        return entity

    @classmethod
    def _drop_analyze_channel(cls, prompt_messages: List[PromptMessage]) -> None:
        """
        Remove thinking content from assistant messages for better performance.

        Uses early exit and pre-compiled regex to minimize overhead.
        Args:
            prompt_messages:

        Returns:

        """
        for p in prompt_messages:
            # Early exit conditions
            if not isinstance(p, AssistantPromptMessage):
                continue
            if not isinstance(p.content, str):
                continue
            # Quick check to avoid regex if not needed
            if not p.content.startswith("<think>"):
                continue

            # Only perform regex substitution when necessary
            new_content = cls._THINK_PATTERN.sub("", p.content, count=1)
            # Only update if changed
            if new_content != p.content:
                p.content = new_content

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        credentials = build_ovh_credentials(credentials)
        schema = self.get_model_schema(model, credentials)
        model_parameters = self._sanitize_model_parameters(schema, dict(model_parameters))
        model_parameters = self._normalize_response_format(model_parameters)
        # OVH strict mode: keep only explicitly allowed API parameters.
        completion_type = LLMMode.value_of(credentials["mode"])
        if completion_type is LLMMode.CHAT:
            allowed_parameters = self._OVH_ALLOWED_CHAT_PARAMETERS
        elif completion_type is LLMMode.COMPLETION:
            allowed_parameters = self._OVH_ALLOWED_COMPLETION_PARAMETERS
        else:
            allowed_parameters = set()
        model_parameters = {k: v for k, v in model_parameters.items() if k in allowed_parameters}
        tool_count = len(tools) if tools else 0
        supports_tool_call = self._supports_tool_call(schema)
        if tools and not supports_tool_call:
            tools = None

        invoke_credentials = dict(credentials)
        if tools and supports_tool_call:
            invoke_credentials.setdefault("function_calling_type", "tool_call")
            invoke_credentials.setdefault("stream_function_calling", "supported")

        # Remove thinking content from assistant messages for better performance.
        with suppress(Exception):
            self._drop_analyze_channel(prompt_messages)

        # Map token parameter name when needed (Responses API style)
        param_pref = credentials.get("token_param_name", "auto")

        def _needs_max_completion_tokens(m: str) -> bool:
            return bool(re.match(r"^(o1|o3|gpt-5)", m, re.IGNORECASE))

        use_max_completion = (
            (param_pref == "max_completion_tokens")
            or (param_pref == "auto" and _needs_max_completion_tokens(model))
        )

        if use_max_completion:
            # Only map if caller didn't already provide max_completion_tokens
            if "max_completion_tokens" not in model_parameters and "max_tokens" in model_parameters:
                model_parameters["max_completion_tokens"] = model_parameters.pop("max_tokens")

        result = super()._invoke(
            model, invoke_credentials, prompt_messages, model_parameters, tools, stop, stream, None
        )
        return result

    @staticmethod
    def _supports_tool_call(schema: AIModelEntity | None) -> bool:
        if schema is None or not schema.features:
            return False
        tool_features = {
            ModelFeature.TOOL_CALL,
            ModelFeature.MULTI_TOOL_CALL,
            ModelFeature.STREAM_TOOL_CALL,
        }
        return any(feature in schema.features for feature in tool_features)

    @staticmethod
    def _build_allowed_parameter_names(schema: AIModelEntity | None) -> set[str]:
        if schema is None:
            return set()
        allowed: set[str] = set()
        for rule in schema.parameter_rules or []:
            if getattr(rule, "name", None):
                allowed.add(rule.name)
        # Keep OpenAI-compatible aliases accepted by some endpoints.
        if "max_tokens" in allowed:
            allowed.add("max_completion_tokens")
        return allowed

    @staticmethod
    def _normalize_response_format(model_parameters: dict) -> dict:
        normalized = dict(model_parameters)
        response_format = normalized.get("response_format")
        json_schema = normalized.get("json_schema")

        if response_format == "json_schema":
            # Dify's LLM block decides whether a JSON schema is available.
            # If none is provided, fall back to plain text instead of forwarding
            # an invalid payload.
            if not json_schema:
                normalized.pop("response_format", None)
                normalized.pop("json_schema", None)
        elif response_format == "text":
            normalized.pop("response_format", None)
            normalized.pop("json_schema", None)
        elif not response_format:
            normalized.pop("json_schema", None)

        return normalized

    def _sanitize_model_parameters(
        self,
        schema: AIModelEntity | None,
        model_parameters: dict,
    ) -> dict:
        allowed = self._build_allowed_parameter_names(schema)
        if not allowed:
            return {}
        return {k: v for k, v in model_parameters.items() if k in allowed}
