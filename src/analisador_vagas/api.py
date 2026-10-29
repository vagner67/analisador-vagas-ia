"""Endpoint FastAPI opcional. Rode: uvicorn analisador_vagas.api:app"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from . import db
from .llm import LLMClient, LLMError, criar_cliente_do_ambiente
from .service import analisar

app = FastAPI(title="Analisador de Vagas IA")


class Entrada(BaseModel):
    texto: str
    skills: list[str] = []
    salvar: bool = True


def get_cliente() -> LLMClient:
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
    if entrada.salvar:
        conn = db.conectar(os.getenv("VAGAS_DB", "vagas.db"))
        try:
            saida["id"] = db.salvar(conn, res, entrada.texto, cliente.nome)
        finally:
            conn.close()
    return saida


@app.get("/relatorios/top-skills")
def relatorio_top_skills(limite: int = 10) -> list[dict]:
    conn = db.conectar(os.getenv("VAGAS_DB", "vagas.db"))
    try:
        return db.top_skills(conn, limite)
    finally:
        conn.close()
