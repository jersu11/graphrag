# Licensed under the MIT License

"""Chat-based Anthropic LLM implementation."""

from collections.abc import Callable
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from graphrag.query.llm.base import BaseLLM, BaseLLMCallback
from graphrag.query.llm.claude.base import ClaudeLLMImpl
from graphrag.query.llm.claude.typing import (
    CLAUDE_RETRY_ERROR_TYPES,
    ClaudeApiType,
)
from graphrag.query.progress import StatusReporter

_MODEL_REQUIRED_MSG = "model is required"


class ChatClaude(BaseLLM, ClaudeLLMImpl):
    """Wrapper for Anthropic ChatCompletion models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        api_type: ClaudeApiType = ClaudeApiType.Anthropic,
        organization: str | None = None,
        max_retries: int = 10,
        request_timeout: float = 180.0,
        retry_error_types: tuple[type[BaseException]] = CLAUDE_RETRY_ERROR_TYPES,  # type: ignore
        reporter: StatusReporter | None = None,
    ):
        ClaudeLLMImpl.__init__(
            self=self,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,  # type: ignore
            organization=organization,
            max_retries=max_retries,
            request_timeout=request_timeout,
            reporter=reporter,
        )
        self.model = model
        self.retry_error_types = retry_error_types

    def generate(
        self,
        messages: str | list[Any],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text."""

        # Extract system message if present
        system_message = None
        filtered_messages = []
        if not isinstance(messages, str):
            for message in messages:
                if message.get('role') == 'system':
                    system_message = message.get('content')
                else:
                    filtered_messages.append(message)

        messages = filtered_messages

        # go for max tokens
        kwargs['extra_headers'] = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}

        # Add system message to args if found
        if system_message:
            kwargs['system'] = system_message

        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    return self._generate(
                        messages=messages,
                        streaming=streaming,
                        callbacks=callbacks,
                        **kwargs,
                    )
        except RetryError as e:
            self._reporter.error(
                message="Error at generate()", details={self.__class__.__name__: str(e)}
            )
            return ""
        else:
            # TODO: why not just throw in this case?
            return ""

    async def agenerate(
        self,
        messages: str | list[Any],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text asynchronously."""

        # Extract system message if present
        system_message = None
        filtered_messages = []
        if not isinstance(messages, str):
            for message in messages:
                if message.get('role') == 'system':
                    system_message = message.get('content')
                else:
                    filtered_messages.append(message)

        messages = filtered_messages

        # go for max tokens
        kwargs['extra_headers'] = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}

        # Add system message to args if found
        if system_message:
            kwargs['system'] = system_message
            
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),  # type: ignore
            )
            async for attempt in retryer:
                with attempt:
                    return await self._agenerate(
                        messages=messages,
                        streaming=streaming,
                        callbacks=callbacks,
                        **kwargs,
                    )
        except RetryError as e:
            self._reporter.error(f"Error at agenerate(): {e}")
            return ""
        else:
            # TODO: why not just throw in this case?
            return ""

    def _generate(
        self,
        messages: str | list[Any],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)

        if streaming:
            full_response = ""
            with self.sync_client.messages.stream(
                messages=messages,
                model=model,
                **kwargs,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    if callbacks:
                        for callback in callbacks:
                            callback.on_llm_new_token(delta)
            return full_response

        else:
            response = self.sync_client.messages.create(
                model=model,
                messages=messages,
                stream=streaming,
                **kwargs,
            ) 
            return response.content[0].text

    async def _agenerate(
        self,
        messages: str | list[Any],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        model = self.model
        if not model:
            raise ValueError(_MODEL_REQUIRED_MSG)

        if streaming:
            full_response = ""
            async with self.async_client.messages.stream(
                messages=messages,
                model=model,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    full_response += text
                    if callbacks:
                        for callback in callbacks:
                            callback.on_llm_new_token(delta)
            return full_response

        else:
            response = await self.async_client.messages.create(
                model=model,
                messages=messages,
                stream=streaming,
                **kwargs,
            ) 
            return response.content[0].text
