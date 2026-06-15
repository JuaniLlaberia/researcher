import os
from typing import Any, AsyncIterator, Dict, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

class LLMConfig(BaseModel):
    """
    Structured object to configure LLM wrapper
    """
    provider: Literal["anthropic", "openai", "google", "ollama"]
    model: str
    temperature: float

class LLM:
    """
    Custom LLM wrapper, that allows you to work with any of the supported providers just by changing the config object.

    Attributes:
        config (LLMConfig): Pydantic object containing the LLM configuration (provider, model and temperature).
        timeout (float): Per-request timeout in seconds; prevents a hung call from blocking the worker.
        max_retries (int): Number of automatic retries on transient errors (timeouts, 429, 5xx).
        llm (ChatAnthropic | ChatOpenAI | ChatGoogleGenerativeAI | ChatOllama): Instance of configured LLM.
    """
    def __init__(
        self,
        config: LLMConfig,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        """
        Initializes LLM class.

        Args:
            config (LLMConfig): Pydantic object containing the LLM configuration (provider, model and temperature).
            timeout (float): Per-request timeout in seconds; prevents a hung call from blocking the worker.
            max_retries (int): Number of automatic retries on transient errors (timeouts, 429, 5xx).
        Returns:
            None.
        """
        self.config = config
        self.timeout = timeout
        self.max_retries = max_retries
        self.llm = self._get_llm()

    def _get_llm(self):
        """
        Generates instances of LLM based on config object.

        Returns:
            (ChatAnthropic | ChatOpenAI | ChatGoogleGenerativeAI | ChatOllama): Instance of model.
        """
        provider = self.config.provider
        model = self.config.model
        temperature = self.config.temperature

        if provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("Missing ANTHROPIC_API_KEY env variable.")

            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature, timeout=self.timeout, max_retries=self.max_retries)
        elif provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("Missing OPENAI_API_KEY env variable.")

            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=temperature, timeout=self.timeout, max_retries=self.max_retries)
        elif provider == "google":
            if not os.getenv("GOOGLE_GEMINI_KEY"):
                raise ValueError("Missing GOOGLE_GEMINI_KEY env variable.")

            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=temperature, timeout=self.timeout, max_retries=self.max_retries)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, temperature=temperature, timeout=self.timeout, max_retries=self.max_retries)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def invoke(
        self,
        prompt: ChatPromptTemplate,
        input: Dict[str, Any],
        output_schema: type[BaseModel] | None = None,
    ) -> BaseModel | BaseMessage:
        """
        Run a prompt through the model.

        When ``output_schema`` is provided, the model is asked to return structured
        output parsed into that schema. Otherwise the raw chat message is returned,
        which is what conversational/free-form callers (Critic, Writing Assistant) want.

        Exceptions are intentionally not caught here: handle them at the graph node,
        where state is available to write a structured failure/recovery record.

        Args:
            prompt (ChatPromptTemplate): Prompt template to format and send.
            input (Dict[str, Any]): Values for the prompt template variables.
            output_schema (type[BaseModel] | None): Pydantic schema for structured output.
        Returns:
            BaseModel: An instance of ``output_schema`` when one is given.
            BaseMessage: The raw model message when no schema is given.
        """
        runnable = self.llm if output_schema is None else self.llm.with_structured_output(output_schema)
        chain = prompt | runnable
        return chain.invoke(input)

    async def astream(
        self,
        prompt: ChatPromptTemplate,
        input: Dict[str, Any],
    ) -> AsyncIterator[BaseMessage]:
        """
        Stream the model response chunk by chunk (free-form text only).

        Intended for the API layer's SSE endpoint. Structured output is not streamed
        because partial schema chunks are not meaningful.

        Args:
            prompt (ChatPromptTemplate): Prompt template to format and send.
            input (Dict[str, Any]): Values for the prompt template variables.
        Yields:
            BaseMessage: Incremental message chunks as they arrive.
        """
        chain = prompt | self.llm
        async for chunk in chain.astream(input):
            yield chunk