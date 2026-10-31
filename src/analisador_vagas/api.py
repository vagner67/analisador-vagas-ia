"""Endpoint FastAPI opcional. Rode: uvicorn analisador_vagas.api:app"""

from __future__ import annotations

import os
from importlib.resources import files

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import db
from .heuristica import HeuristicaClient
from .llm import LLMClient, LLMError, criar_cliente_do_ambiente
from .service import analisar

app = FastAPI(title="Analisador de Vagas IA")


class Entrada(BaseModel):
    texto: str
    skills: list[str] = []
    salvar: bool = True


def _db_path() -> str:
    padrao = "/tmp/vagas.db" if os.getenv("VERCEL") else "vagas.db"
    return os.getenv("VAGAS_DB", padrao)


def _conectar():
    """Abre o banco; na demo Vercel (ou DEMO_SEED=1) popula vagas de exemplo se vazio.

    As vagas de exemplo são escritas à mão (ilustrativas) e analisadas no modo sem LLM.
    """
    conn = db.conectar(_db_path())
    if (os.getenv("VERCEL") or os.getenv("DEMO_SEED") == "1") and not conn.execute(
        "SELECT 1 FROM vagas LIMIT 1"
    ).fetchone():
        import json

        texto = files("analisador_vagas").joinpath("dados/vagas_exemplo.jsonl").read_text("utf-8")
        cli = HeuristicaClient()
        for linha in texto.splitlines():
            if linha.strip():
                t = json.loads(linha)["texto"]
                db.salvar(conn, analisar(t, cli, []), t, cli.nome + " [exemplo]")
    return conn


def _tem_chave() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))


def get_cliente() -> LLMClient:
    if not _tem_chave():
        return HeuristicaClient()
    try:
        return criar_cliente_do_ambiente()
    except (LLMError, httpx.HTTPError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/analisar")
def analisar_vaga(entrada: Entrada, cliente: LLMClient = Depends(get_cliente)) -> dict:
    try:
        res = analisar(entrada.texto, cliente, entrada.skills)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except (LLMError, httpx.HTTPError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    saida = res.to_dict()
    saida["modo"] = cliente.nome
    saida["usa_llm"] = not isinstance(cliente, HeuristicaClient)
    if entrada.salvar:
        conn = _conectar()
        try:
            saida["id"] = db.salvar(conn, res, entrada.texto, cliente.nome)
        finally:
            conn.close()
    return saida


class EntradaLote(BaseModel):
    textos: list[str] = Field(min_length=1, max_length=20)
    skills: list[str] = []
    concorrencia: int = Field(default=4, ge=1, le=8)


@app.post("/analisar/lote")
async def analisar_lote_endpoint(
    entrada: EntradaLote, cliente: LLMClient = Depends(get_cliente)
) -> dict:
    """Analisa até 20 vagas concorrentemente (asyncio + semáforo). Não salva no banco."""
    from .assincrono import analisar_lote

    itens = await analisar_lote(entrada.textos, cliente, entrada.skills, entrada.concorrencia)
    return {
        "modo": cliente.nome,
        "usa_llm": not isinstance(cliente, HeuristicaClient),
        "resultados": [i.to_dict() for i in itens],
    }


@app.get("/relatorios/top-skills")
def relatorio_top_skills(limite: int = 10) -> list[dict]:
    conn = _conectar()
    try:
        return db.top_skills(conn, limite)
    finally:
        conn.close()


class Pergunta(BaseModel):
    pergunta: str
    k: int = 3


@app.post("/perguntar")
def perguntar(entrada: Pergunta) -> dict:
    """RAG sobre as vagas salvas; resposta cita ids em `fontes`."""
    from . import rag

    conn = _conectar()
    try:
        emb, llm_ = rag.componentes_do_ambiente()
        return rag.perguntar(conn, entrada.pergunta, entrada.k, emb, llm_).__dict__
    except (LLMError, httpx.HTTPError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    finally:
        conn.close()


@app.post("/agente")
def agente_endpoint(entrada: Pergunta) -> dict:
    """Agente com tool-calling (buscar_vagas, calcular_match, gerar_relatorio_sql)."""
    from . import agente
    from .embeddings import embedder_do_ambiente

    conn = _conectar()
    try:
        return agente.rodar(entrada.pergunta, conn, embedder_do_ambiente())
    except (LLMError, httpx.HTTPError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    finally:
        conn.close()


@app.get("/", response_class=HTMLResponse)
def pagina() -> str:
    return files("analisador_vagas").joinpath("static/index.html").read_text("utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modo": "llm" if _tem_chave() else "modo sem LLM"}
