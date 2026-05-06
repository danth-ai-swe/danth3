"""Azure OpenAI client factories.

Single source of truth — replaces the duplicated factories in chat.py,
eval.py, and index.py.
"""
from __future__ import annotations

from openai import AsyncAzureOpenAI, AzureOpenAI

from loma_rag.config.settings import azure


def make_chat_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=azure.api_base.rstrip("/"),
        api_key=azure.api_key,
        api_version=azure.api_version,
    )


def make_async_chat_client() -> AsyncAzureOpenAI:
    """AsyncAzureOpenAI client — used for parallel LLM calls (analyze_query
    parallel with embed, async streaming endpoints, etc.)."""
    return AsyncAzureOpenAI(
        azure_endpoint=azure.api_base.rstrip("/"),
        api_key=azure.api_key,
        api_version=azure.api_version,
    )


# Dense embeddings use the same Azure deployment under a different model name;
# reusing the same client class is fine.
def make_dense_client() -> AzureOpenAI:
    return make_chat_client()
