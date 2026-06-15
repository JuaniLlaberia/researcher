import os
from typing import  Literal, List
from pydantic import BaseModel

class EmbedderConfig(BaseModel):
    """
    Structured object to configure embedder wrapper
    """
    provider: Literal["openai", "google", "ollama"]
    model: str

class Embedder:
    """
    Custom embedder wrapper, that allows you to work with any of the supported providers just by changing the config object.

    Attributes:
        config (EmbedderConfig): Pydantic object containing the embedder configuration (provider and model).
        embedder (OpenAIEmbeddings | GoogleGenerativeAIEmbeddings | OllamaEmbeddings): Instance of configured embedder.
    """
    def __init__(self, config: EmbedderConfig) -> None:
        """
        Initializes embedder class.

        Args:
            config (LLMConfig): Pydantic object containing the LLM configuration (provider and model).
        Returns:
            None.
        """
        self.config = config
        self.embedder = self._get_embedder()

    def _get_embedder(self):
        """
        Generates instances of embedder based on config object.

        Returns:
            (OpenAIEmbeddings | GoogleGenerativeAIEmbeddings | OllamaEmbeddings): Instance of model.
        """
        provider = self.config.provider
        model = self.config.model

        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("Missing OPENAI_API_KEY env variable.")

            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model)
        elif provider == "google":
            if not os.getenv("GOOGLE_GEMINI_KEY"):
                raise ValueError("Missing GOOGLE_GEMINI_KEY env variable.")

            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=model)
        elif provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model=model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for provided texts.

        Args:
            texts (List[str]): List of texts to embed.
        Returns:
            List[List[float]]: List of embeddings.
        """
        return self.embedder.embed_documents(texts=texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generates embedding for query.

        Args:
            text (str): Query to embed.
        Returns:
            List[float]: Query embedding.
        """
        return self.embedder.embed_query(text)