# Licensed under the MIT License

"""Base classes for LLM and Embedding models."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from anthropic import AsyncAnthropic, Anthropic

from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.claude.typing import ClaudeApiType
from graphrag.query.progress import ConsoleStatusReporter, StatusReporter


class BaseClaudeLLM(ABC):
    """The Base Claude LLM implementation."""

    _async_client: AsyncAnthropic
    _sync_client: Anthropic

    def __init__(self):
        self._create_claude_client()

    @abstractmethod
    def _create_claude_client(self):
        """Create a new synchronous and asynchronous Claude client instance."""

    def set_clients(
        self,
        sync_client: Anthropic,
        async_client: AsyncAnthropic,
    ):
        """
        Set the synchronous and asynchronous clients used for making API requests.

        Args:
            sync_client (Anthropic): The sync client object.
            async_client (AsyncAnthropic): The async client object.
        """
        self._sync_client = sync_client
        self._async_client = async_client

    @property
    def async_client(self) -> AsyncAnthropic | None:
        """
        Get the asynchronous client used for making API requests.

        Returns
        -------
            AsynAnthropic: The async client object.
        """
        return self._async_client

    @property
    def sync_client(self) -> Anthropic | None:
        """
        Get the synchronous client used for making API requests.

        Returns
        -------
            Anthropic: The async client object.
        """
        return self._sync_client

    @async_client.setter
    def async_client(self, client: AsyncAnthropic):
        """
        Set the asynchronous client used for making API requests.

        Args:
            client (AsyncAnthropic): The async client object.
        """
        self._async_client = client

    @sync_client.setter
    def sync_client(self, client: Anthropic):
        """
        Set the synchronous client used for making API requests.

        Args:
            client (Anthropic): The sync client object.
        """
        self._sync_client = client


class ClaudeLLMImpl(BaseClaudeLLM):
    """Orchestration Claude LLM Implementation."""

    _reporter: StatusReporter = ConsoleStatusReporter()

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        api_type: ClaudeApiType = ClaudeApiType.Anthropic,
        organization: str | None = None,
        max_retries: int = 10,
        request_timeout: float = 180.0,
        reporter: StatusReporter | None = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.api_type = api_type
        self.organization = organization
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.reporter = reporter or ConsoleStatusReporter()

        try:
            # Create Anthropic sync and async clients
            super().__init__()
        except Exception as e:
            self._reporter.error(
                message="Failed to create Anthropic client",
                details={self.__class__.__name__: str(e)},
            )
            raise

    def _create_claude_client(self):
        """Create a new Anthropic client instance."""
        sync_client = Anthropic(
            api_key=self.api_key,
            base_url=self.api_base,
            # Retry Configuration
            timeout=self.request_timeout,
            max_retries=self.max_retries,
        )

        async_client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.api_base,
            # Retry Configuration
            timeout=self.request_timeout,
            max_retries=self.max_retries,
        )
        self.set_clients(sync_client=sync_client, async_client=async_client)
