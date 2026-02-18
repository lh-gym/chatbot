"""Configuration models for the RAG system."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """Configures semantic + sliding-window chunking behavior."""

    min_tokens: int = Field(default=500, ge=100)
    max_tokens: int = Field(default=1200, ge=200)
    overlap_tokens: int = Field(default=120, ge=20)
    semantic_break_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class RetrievalConfig(BaseModel):
    """Configures dual-route retrieval and fusion heuristics."""

    semantic_k: int = Field(default=20, ge=1)
    metadata_k: int = Field(default=20, ge=1)
    final_k: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)


class AgentConfig(BaseModel):
    """Configures agent execution and latency/cost targets."""

    max_iterations: int = Field(default=6, ge=1)
    target_latency_seconds: float = Field(default=8.0, gt=0.0)
    groundedness_target: float = Field(default=0.95, ge=0.0, le=1.0)
