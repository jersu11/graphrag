# Licensed under the MIT License

"""A text-completion based LLM."""

import logging

from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
)

from .claude_configuration import ClaudeConfiguration
from .types import ClaudeClientTypes
from .utils import get_completion_llm_args

log = logging.getLogger(__name__)


class ClaudeCompletionLLM(BaseLLM[CompletionInput, CompletionOutput]):
    """A text-completion based LLM."""

    _client: ClaudeClientTypes
    _configuration: ClaudeConfiguration

    def __init__(self, client: ClaudeClientTypes, configuration: ClaudeConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput | None:
        args = get_completion_llm_args(
            kwargs.get("model_parameters"), self.configuration
        )
        completion = await self.client.completions.create(prompt=input, **args)
        return completion.content
