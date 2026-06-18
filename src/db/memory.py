from typing import List
from langchain_core.messages import BaseMessage

from src.core.config import settings
from src.core.llm import LLM, LLMConfig
from .utils.prompts import SUMMARY_PROMPT
from .utils.models import SummaryOutput

def generate_summary_memory(crr_summary: str, messages: List[BaseMessage]) -> str:
    """
    Generates summary for a session based on current summary and the window of messages.

    Args:
        crr_summary (str): Session previous summary.
        messages (List[BaseMessage]): Messages in context window.
    Returns:
        str: Generated summary.
    """
    llm = LLM(config=LLMConfig(
        provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.llm_temperature
    ))
    
    result = llm.invoke(
        prompt=SUMMARY_PROMPT,
        input={
            "crr_summary": crr_summary,
            "messages": messages
        },
        output_schema=SummaryOutput
    )

    return result if isinstance(result, SummaryOutput) else SummaryOutput(**result.model_dump())