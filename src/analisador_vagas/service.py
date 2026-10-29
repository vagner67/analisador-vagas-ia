"""Orquestra LLM + cálculo de match."""

from __future__ import annotations

from .llm import LLMClient
from .models import AnaliseVaga, calcular_match


def analisar(texto: str, cliente: LLMClient, minhas_skills: list[str]) -> AnaliseVaga:
    if not texto.strip():
        raise ValueError("Texto da vaga vazio.")
    analise = AnaliseVaga.from_dict(cliente.extrair(texto))
    analise.match_score = calcular_match(analise, minhas_skills)
    return analise
