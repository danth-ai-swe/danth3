"""Docx ingestion entrypoint — re-exports chunking primitives."""
from loma_rag.chunking.chunker import chunk_docx

__all__ = ["chunk_docx"]
