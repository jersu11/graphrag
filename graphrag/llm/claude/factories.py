# Licensed under the MIT License

"""Factory functions for creating Claude LLMs."""

import asyncio

from graphrag.llm.base import CachingLLM, RateLimitingLLM
from graphrag.llm.limiting import LLMLimiter
from graphrag.llm.types import (
    LLM,
    CompletionLLM,
    EmbeddingLLM,
    ErrorHandlerFn,
    LLMCache,
    LLMInvocationFn,
    OnCacheActionFn,
)

from .json_parsing_llm import JsonParsingLLM
from .claude_chat_llm import ClaudeChatLLM
from .claude_completion_llm import ClaudeCompletionLLM
from .claude_configuration import ClaudeConfiguration
from .claude_history_tracking_llm import ClaudeHistoryTrackingLLM
from .claude_token_replacing_llm import ClaudeTokenReplacingLLM
from .types import ClaudeClientTypes
from .utils import (
    RATE_LIMIT_ERRORS,
    RETRYABLE_ERRORS,
    get_completion_cache_args,
    get_sleep_time_from_error,
    get_token_counter,
)


def create_claude_chat_llm(
    client: ClaudeClientTypes,
    config: ClaudeConfiguration,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> CompletionLLM:
    """Create an Claude chat LLM."""
    operation = "chat"
    result = ClaudeChatLLM(client, config)
    result.on_error(on_error)
    if limiter is not None or semaphore is not None:
        result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)
    if cache is not None:
        result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)
    result = ClaudeHistoryTrackingLLM(result)
    result = ClaudeTokenReplacingLLM(result)
    return JsonParsingLLM(result)


def create_claude_completion_llm(
    client: ClaudeClientTypes,
    config: ClaudeConfiguration,
    cache: LLMCache | None = None,
    limiter: LLMLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_invoke: LLMInvocationFn | None = None,
    on_error: ErrorHandlerFn | None = None,
    on_cache_hit: OnCacheActionFn | None = None,
    on_cache_miss: OnCacheActionFn | None = None,
) -> CompletionLLM:
    """Create an Claude completion LLM."""
    operation = "completion"
    result = ClaudeCompletionLLM(client, config)
    result.on_error(on_error)
    if limiter is not None or semaphore is not None:
        result = _rate_limited(result, config, operation, limiter, semaphore, on_invoke)
    if cache is not None:
        result = _cached(result, config, operation, cache, on_cache_hit, on_cache_miss)
    return ClaudeTokenReplacingLLM(result)


def _rate_limited(
    delegate: LLM,
    config: ClaudeConfiguration,
    operation: str,
    limiter: LLMLimiter | None,
    semaphore: asyncio.Semaphore | None,
    on_invoke: LLMInvocationFn | None,
):
    result = RateLimitingLLM(
        delegate,
        config,
        operation,
        RETRYABLE_ERRORS,
        RATE_LIMIT_ERRORS,
        limiter,
        semaphore,
        get_token_counter(config),
        get_sleep_time_from_error,
    )
    result.on_invoke(on_invoke)
    return result


def _cached(
    delegate: LLM,
    config: ClaudeConfiguration,
    operation: str,
    cache: LLMCache,
    on_cache_hit: OnCacheActionFn | None,
    on_cache_miss: OnCacheActionFn | None,
):
    cache_args = get_completion_cache_args(config)
    result = CachingLLM(delegate, cache_args, operation, cache)
    result.on_cache_hit(on_cache_hit)
    result.on_cache_miss(on_cache_miss)
    return result
