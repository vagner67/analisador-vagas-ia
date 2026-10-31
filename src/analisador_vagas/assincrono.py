"""Análise concorrente de várias vagas com asyncio.

Os clientes LLM usam `httpx` síncrono (bloqueante). Em vez de duplicar cada cliente em versão
async, cada chamada roda em uma thread via `asyncio.to_thread`, e um `asyncio.Semaphore`
limita quantas chamadas ficam em voo ao mesmo tempo (respeita rate limit do provedor).
Falhas são isoladas por vaga: uma vaga com erro não derruba o lote.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .llm import LLMClient
from .models import AnaliseVaga
from .service import analisar


@dataclass
class ItemLote:
    indice: int
    analise: AnaliseVaga | None = None
    erro: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"indice": self.indice}
        if self.analise is not None:
            d["analise"] = self.analise.to_dict()
        if self.erro is not None:
            d["erro"] = self.erro
        return d


async def analisar_async(texto: str, cliente: LLMClient, skills: list[str]) -> AnaliseVaga:
    """Versão não bloqueante de `service.analisar` (roda em thread do executor padrão)."""
    return await asyncio.to_thread(analisar, texto, cliente, skills)


async def analisar_lote(
    textos: list[str],
    cliente: LLMClient,
    skills: list[str],
    concorrencia: int = 4,
    timeout: float | None = None,
) -> list[ItemLote]:
    """Analisa `textos` concorrentemente; resultado na mesma ordem da entrada."""
    if concorrencia < 1:
        raise ValueError("concorrencia deve ser >= 1")
    sem = asyncio.Semaphore(concorrencia)

    async def um(i: int, texto: str) -> ItemLote:
        async with sem:
            try:
                coro = analisar_async(texto, cliente, skills)
                analise = await (asyncio.wait_for(coro, timeout) if timeout else coro)
                return ItemLote(i, analise=analise)
            except asyncio.TimeoutError:
                return ItemLote(i, erro=f"timeout após {timeout}s")
            except Exception as e:  # noqa: BLE001 - isolamos falhas por item
                return ItemLote(i, erro=f"{type(e).__name__}: {e}")

    return list(await asyncio.gather(*(um(i, t) for i, t in enumerate(textos))))
