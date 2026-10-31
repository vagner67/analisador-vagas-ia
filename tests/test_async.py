import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from analisador_vagas.assincrono import analisar_async, analisar_lote
from analisador_vagas.heuristica import HeuristicaClient

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class LentoLLM:
    """Cliente fake bloqueante que mede quantas chamadas rodam ao mesmo tempo."""

    nome = "lento"

    def __init__(self, atraso: float = 0.1, falhar_em: str | None = None) -> None:
        self.atraso, self.falhar_em = atraso, falhar_em
        self.ativos = self.pico = 0
        self._lock = threading.Lock()

    def extrair(self, texto: str) -> dict:
        with self._lock:
            self.ativos += 1
            self.pico = max(self.pico, self.ativos)
        try:
            time.sleep(self.atraso)
            if self.falhar_em and self.falhar_em in texto:
                raise RuntimeError("falha simulada")
            return {
                "titulo": texto,
                "senioridade": "junior",
                "skills_obrigatorias": ["python"],
                "skills_desejaveis": [],
            }
        finally:
            with self._lock:
                self.ativos -= 1


async def test_analisar_async_equivale_ao_sincrono():
    a = await analisar_async("Dev Python com FastAPI e SQL", HeuristicaClient(), ["python"])
    assert "python" in a.skills_obrigatorias + a.skills_desejaveis


async def test_lote_roda_concorrente_e_preserva_ordem():
    cli = LentoLLM(atraso=0.2)
    t0 = time.perf_counter()
    itens = await analisar_lote([f"vaga {i}" for i in range(4)], cli, ["python"], concorrencia=4)
    dur = time.perf_counter() - t0
    assert [i.analise.titulo for i in itens] == [f"vaga {i}" for i in range(4)]
    assert cli.pico == 4
    assert dur < 0.6  # sequencial levaria >= 0.8s


async def test_semaforo_limita_concorrencia():
    cli = LentoLLM(atraso=0.05)
    await analisar_lote([f"v{i}" for i in range(6)], cli, [], concorrencia=2)
    assert cli.pico == 2


async def test_falha_isolada_por_item():
    itens = await analisar_lote(["ok 1", "BOOM", "ok 2"], LentoLLM(0.01, "BOOM"), [])
    assert itens[0].analise and itens[2].analise
    assert itens[1].analise is None and "falha simulada" in itens[1].erro
    vazio = await analisar_lote(["   "], LentoLLM(0.0), [])
    assert "ValueError" in vazio[0].erro


async def test_timeout_por_item():
    itens = await analisar_lote(["x"], LentoLLM(atraso=0.5), [], timeout=0.05)
    assert "timeout" in itens[0].erro


async def test_concorrencia_invalida():
    with pytest.raises(ValueError):
        await analisar_lote(["x"], LentoLLM(0.0), [], concorrencia=0)


def test_endpoint_lote(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from analisador_vagas.api import app

    c = TestClient(app)
    r = c.post(
        "/analisar/lote",
        json={"textos": ["Python e Docker", "React e TypeScript"], "skills": ["python"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["usa_llm"] is False and len(body["resultados"]) == 2
    assert [x["indice"] for x in body["resultados"]] == [0, 1]
    assert c.post("/analisar/lote", json={"textos": []}).status_code == 422


def test_sem_loop_bloqueado():
    # garante que a corrotina é usável fora do pytest (asyncio.run)
    itens = asyncio.run(analisar_lote(["Python"], HeuristicaClient(), []))
    assert itens[0].erro is None
