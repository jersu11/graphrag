# Licensed under the MIT License

"""Anthropic wrapper options."""

from enum import Enum
from typing import Any, cast

import anthropic

CLAUDE_RETRY_ERROR_TYPES = (
    cast(Any, anthropic).RateLimitError,
    cast(Any, anthropic).APIConnectionError,
)


class ClaudeApiType(str, Enum):
    """The Anthropic Flavor."""

    Anthropic = "claude"
