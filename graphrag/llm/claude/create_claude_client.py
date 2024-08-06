# Licensed under the MIT License

"""Create Claude client instance."""

import logging
from functools import cache

from anthropic import AsyncAnthropic

from .claude_configuration import ClaudeConfiguration
from .types import ClaudeClientTypes

log = logging.getLogger(__name__)


@cache
def create_claude_client(
    configuration: ClaudeConfiguration
) -> ClaudeClientTypes:
    """Create a new Claude client instance."""
    log.info("Creating Claude client base_url=%s", configuration.api_base)
    return AsyncAnthropic(
        api_key=configuration.api_key,
        base_url=configuration.api_base,
        # Timeout/Retry Configuration - Use Tenacity for Retries, so disable them here
        timeout=configuration.request_timeout or 180.0,
        max_retries=0,
    )
