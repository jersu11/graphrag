# Licensed under the MIT License

"""The Chat-based language model."""

import logging
from json import JSONDecodeError

from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
    LLMOutput,
)

from ._json import clean_up_json
from .claude_configuration import ClaudeConfiguration
from .types import ClaudeClientTypes
from .utils import (
    get_completion_llm_args,
    try_parse_json_object,
)

log = logging.getLogger(__name__)

_MAX_GENERATION_RETRIES = 3
FAILED_TO_CREATE_JSON_ERROR = "Failed to generate valid JSON output"


class ClaudeChatLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A Chat-based LLM."""

    _client: ClaudeClientTypes
    _configuration: ClaudeConfiguration

    def __init__(self, client: ClaudeClientTypes, configuration: ClaudeConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    ) -> CompletionOutput | None:
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.configuration
        )
        history = kwargs.get("history") or []

        # Extract system message if present
        system_message = None
        filtered_history = []
        for message in history:
            if message.get('role') == 'system':
                system_message = message.get('content')
            else:
                filtered_history.append(message)

        messages = [
            *filtered_history,
            {"role": "user", "content": input},
        ]

        # Add system message to args if found
        if system_message:
            args['system'] = system_message

        response = await self.client.messages.create(
            messages=messages, **args
        )
        return response.content[0].text


    async def _invoke_json(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> LLMOutput[CompletionOutput]:
        """Generate JSON output."""
        name = kwargs.get("name") or "unknown"
        is_response_valid = kwargs.get("is_response_valid") or (lambda _x: True)

        async def generate(
            attempt: int | None = None,
        ) -> LLMOutput[CompletionOutput]:
            call_name = name if attempt is None else f"{name}@{attempt}"
            
            # Create a copy of kwargs to avoid modifying the original
            modified_kwargs = kwargs.copy()
            
            # If this is the second attempt (attempt == 1), update the model
            if attempt > 0:
                if self.configuration.model_alt:
                    new_model = self.configuration.model_alt
                    modified_kwargs['model'] = new_model
            
            # Update the name in the kwargs
            modified_kwargs['name'] = call_name

            return (
                await self._native_json(input, **modified_kwargs)
                if self.configuration.model_supports_json
                else await self._manual_json(input, **modified_kwargs)
            )

        def is_valid(x: dict | None) -> bool:
            return x is not None and is_response_valid(x)

        result = await generate(0)
        retry = 1
        while not is_valid(result.json) and retry < _MAX_GENERATION_RETRIES:
            log.info(f"_invoke_json attempt {retry} failed to produce valid JSON. Retrying...")
            result = await generate(retry)
            retry += 1

        if is_valid(result.json):
            return result
        raise RuntimeError(FAILED_TO_CREATE_JSON_ERROR)

    async def _native_json(
        self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        """Generate JSON output using a model's native JSON-output support."""
        result = await self._invoke(
            input,
            **{
                **kwargs,
                "model_parameters": {
                    **(kwargs.get("model_parameters") or {}),
                    "response_format": {"type": "json_object"},
                },
            },
        )

        raw_output = result.output or ""
        json_output = try_parse_json_object(raw_output)

        return LLMOutput[CompletionOutput](
            output=raw_output,
            json=json_output,
            history=result.history,
        )

    async def _manual_json(
        self, input: CompletionInput, **kwargs: Unpack[LLMInput]
    ) -> LLMOutput[CompletionOutput]:
        # Otherwise, clean up the output and try to parse it as json
        result = await self._invoke(input, **kwargs)
        history = result.history or []
        output = clean_up_json(result.output or "")
        try:
            json_output = try_parse_json_object(output)
            return LLMOutput[CompletionOutput](
                output=output, json=json_output, history=history
            )
        except (TypeError, JSONDecodeError):
            # log.warning("error parsing llm json, retrying")
            return LLMOutput[CompletionOutput](
                output="",
                json="",
                history=history,
            )
