"""Endpoint FastAPI opcional. Rode: uvicorn analisador_vagas.api:app"""

from __future__ import annotations

import os
from importlib.resources import files

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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


def _tem_chave() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))


def get_cliente() -> LLMClient:
    if not _tem_chave():
        return HeuristicaClient()
    try:
        return criar_cliente_do_ambiente()
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/analisar")
def analisar_vaga(entrada: Entrada, cliente: LLMClient = Depends(get_cliente)) -> dict:
    try:
        res = analisar(entrada.texto, cliente, entrada.skills)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    saida = res.to_dict()
    saida["modo"] = cliente.nome
    saida["usa_llm"] = not isinstance(cliente, HeuristicaClient)
    if entrada.salvar:
        conn = db.conectar(_db_path())
        try:
            saida["id"] = db.salvar(conn, res, entrada.texto, cliente.nome)
        finally:
            conn.close()
    return saida


@app.get("/relatorios/top-skills")
def relatorio_top_skills(limite: int = 10) -> list[dict]:
    conn = db.conectar(_db_path())
    try:
        return db.top_skills(conn, limite)
    finally:
        conn.close()


@app.get("/", response_class=HTMLResponse)
def pagina() -> str:
    return files("analisador_vagas").joinpath("static/index.html").read_text("utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modo": "llm" if _tem_chave() else "modo sem LLM"}
