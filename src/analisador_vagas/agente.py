"""Agente com tool-calling (OpenAI ou Gemini function calling) sobre as vagas.

Ferramentas: `buscar_vagas` (RAG), `calcular_match`, `gerar_relatorio_sql`.
Sem chave de API, usa um **roteador determinístico offline** (regex, sem LLM), rotulado
em `modo`. Todas as chamadas de ferramenta ficam registradas em `passos`.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Any

import httpx

from . import db, rag
from .embeddings import Embedder
from .llm import LLMError
from .models import AnaliseVaga, calcular_match

MODO_OFFLINE = "offline: roteador determinístico por regex (sem LLM)"
MAX_PASSOS = 5

TOOLS: list[dict] = [
    {
        "name": "buscar_vagas",
        "description": "Busca semântica (RAG) nas vagas analisadas. Retorna ids, títulos e skills.",
        "parameters": {
            "type": "object",
            "properties": {
                "consulta": {"type": "string", "description": "texto da busca"},
                "k": {"type": "integer", "description": "nº de vagas (padrão 3)"},
            },
            "required": ["consulta"],
        },
    },
    {
        "name": "calcular_match",
        "description": "Calcula o match (0-100) entre uma vaga salva e uma lista de skills.",
        "parameters": {
            "type": "object",
            "properties": {
                "vaga_id": {"type": "integer"},
                "skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["vaga_id", "skills"],
        },
    },
    {
        "name": "gerar_relatorio_sql",
        "description": "Relatórios SQL agregados: skills mais pedidas, vagas por senioridade "
        "ou ranking de vagas por match.",
        "parameters": {
            "type": "object",
            "properties": {
                "tipo": {"type": "string", "enum": ["top_skills", "senioridade", "vagas"]},
                "limite": {"type": "integer"},
            },
            "required": ["tipo"],
        },
    },
]


class Ferramentas:
    def __init__(self, conn: sqlite3.Connection, embedder: Embedder | None = None) -> None:
        self.conn, self.embedder = conn, embedder

    def buscar_vagas(self, consulta: str, k: int = 3) -> dict:
        hits, modo = rag.buscar(self.conn, consulta, int(k), self.embedder)
        for h in hits:
            h.pop("trecho", None)
        return {"vagas": hits, "recuperacao": modo}

    def calcular_match(self, vaga_id: int, skills: list[str]) -> dict:
        v = self.conn.execute("SELECT titulo, senioridade FROM vagas WHERE id = ?", (int(vaga_id),))
        row = v.fetchone()
        if row is None:
            return {"erro": f"vaga {vaga_id} não encontrada"}
        sk = self.conn.execute("SELECT skill, tipo FROM vaga_skills WHERE vaga_id = ?", (vaga_id,))
        obr, des = [], []
        for r in sk:
            (obr if r["tipo"] == "obrigatoria" else des).append(r["skill"])
        a = AnaliseVaga(row["titulo"], row["senioridade"], obr, des)
        meus = {s.strip().lower() for s in skills}
        return {
            "vaga_id": int(vaga_id),
            "titulo": row["titulo"],
            "match_score": calcular_match(a, skills),
            "faltando_obrigatorias": [s for s in obr if s not in meus],
        }

    def gerar_relatorio_sql(self, tipo: str, limite: int = 10) -> dict:
        if tipo == "top_skills":
            return {"linhas": db.top_skills(self.conn, int(limite))}
        if tipo == "senioridade":
            return {"linhas": db.por_senioridade(self.conn)}
        if tipo == "vagas":
            return {"linhas": db.listar(self.conn, int(limite))}
        return {"erro": f"tipo inválido: {tipo}"}

    def executar(self, nome: str, args: dict) -> dict:
        if nome not in {t["name"] for t in TOOLS}:
            return {"erro": f"ferramenta desconhecida: {nome}"}
        try:
            return getattr(self, nome)(**args)
        except (TypeError, ValueError) as e:
            return {"erro": f"argumentos inválidos: {e}"}


SISTEMA = (
    "Você é um assistente de carreira. Use as ferramentas para consultar as vagas salvas. "
    "Nunca invente vagas; cite ids no formato [#id]. Responda em português."
)


# ---------------------------------------------------------------- roteador offline
def _skills_na_pergunta(p: str) -> list[str]:
    m = re.search(r"(?:skills?|sei|conheço|conheco|tenho)\s*[:=]?\s*(.+)$", p, re.I)
    if not m:
        return []
    return [s.strip(" .?") for s in re.split(r",|\s+e\s+", m.group(1)) if s.strip(" .?")]


def rotear_offline(pergunta: str) -> list[tuple[str, dict]]:
    """Plano determinístico de chamadas de ferramenta para a pergunta."""
    p = pergunta.lower()
    if re.search(r"senioridade|n[ií]vel", p):
        return [("gerar_relatorio_sql", {"tipo": "senioridade"})]
    if re.search(r"mais (pedid|requisitad|comuns|frequentes)|top skills|ranking de skills", p):
        return [("gerar_relatorio_sql", {"tipo": "top_skills", "limite": 10})]
    if re.search(r"match|compat[ií]ve|compatibilidade|combina", p):
        skills = _skills_na_pergunta(pergunta)
        m = re.search(r"vaga\s*#?(\d+)", p)
        if m:
            return [("calcular_match", {"vaga_id": int(m.group(1)), "skills": skills})]
        return [
            ("buscar_vagas", {"consulta": pergunta, "k": 3}),
            ("calcular_match*", {"skills": skills}),
        ]
    return [("buscar_vagas", {"consulta": pergunta, "k": 3})]


def _rodar_offline(pergunta: str, f: Ferramentas) -> dict:
    passos: list[dict] = []
    for nome, args in rotear_offline(pergunta):
        if nome == "calcular_match*":  # match para cada vaga recuperada no passo anterior
            for v in passos[-1]["resultado"]["vagas"]:
                a = {"vaga_id": v["id"], "skills": args["skills"]}
                passos.append(
                    {
                        "ferramenta": "calcular_match",
                        "args": a,
                        "resultado": f.executar("calcular_match", a),
                    }
                )
            continue
        passos.append({"ferramenta": nome, "args": args, "resultado": f.executar(nome, args)})
    return {"resposta": _resumo(passos), "passos": passos, "modo": MODO_OFFLINE}


def _resumo(passos: list[dict]) -> str:
    linhas = []
    for p in passos:
        r = p["resultado"]
        if "erro" in r:
            linhas.append(f"Erro: {r['erro']}")
        elif p["ferramenta"] == "buscar_vagas":
            linhas += [f"[#{v['id']}] {v['titulo']} — {', '.join(v['skills'])}" for v in r["vagas"]]
            if not r["vagas"]:
                linhas.append("Nenhuma vaga relevante encontrada.")
        elif p["ferramenta"] == "calcular_match":
            falt = ", ".join(r["faltando_obrigatorias"]) or "nenhuma"
            linhas.append(f"[#{r['vaga_id']}] match {r['match_score']} — faltam: {falt}")
        else:
            linhas += [json.dumps(x, ensure_ascii=False) for x in r["linhas"]]
    return "\n".join(linhas)


# ---------------------------------------------------------------- OpenAI
def _rodar_openai(pergunta: str, f: Ferramentas, key: str, model: str) -> dict:
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": pergunta},
    ]
    tools = [{"type": "function", "function": t} for t in TOOLS]
    passos: list[dict] = []
    for _ in range(MAX_PASSOS):
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": model, "temperature": 0, "messages": msgs, "tools": tools},
            headers={"Authorization": f"Bearer {key}"},
            timeout=60,
        )
        if r.status_code != 200:
            raise LLMError(f"OpenAI HTTP {r.status_code}: {r.text[:300]}")
        msg = r.json()["choices"][0]["message"]
        msgs.append(msg)
        if not msg.get("tool_calls"):
            return {
                "resposta": msg.get("content") or "",
                "passos": passos,
                "modo": f"openai:{model}",
            }
        for tc in msg["tool_calls"]:
            nome = tc["function"]["name"]
            args = json.loads(tc["function"].get("arguments") or "{}")
            res = f.executar(nome, args)
            passos.append({"ferramenta": nome, "args": args, "resultado": res})
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(res, ensure_ascii=False),
                }
            )
    raise LLMError("Agente excedeu o limite de passos.")


# ---------------------------------------------------------------- Gemini
def _rodar_gemini(pergunta: str, f: Ferramentas, key: str, model: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": pergunta}]}]
    passos: list[dict] = []
    for _ in range(MAX_PASSOS):
        body = {
            "systemInstruction": {"parts": [{"text": SISTEMA}]},
            "contents": contents,
            "tools": [{"functionDeclarations": TOOLS}],
            "generationConfig": {"temperature": 0},
        }
        r = httpx.post(url, json=body, headers={"x-goog-api-key": key}, timeout=60)
        if r.status_code != 200:
            raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
        content = r.json()["candidates"][0]["content"]
        contents.append(content)
        chamadas = [p["functionCall"] for p in content["parts"] if "functionCall" in p]
        if not chamadas:
            texto = "".join(p.get("text", "") for p in content["parts"])
            return {"resposta": texto, "passos": passos, "modo": f"gemini:{model}"}
        respostas = []
        for c in chamadas:
            res = f.executar(c["name"], c.get("args") or {})
            passos.append({"ferramenta": c["name"], "args": c.get("args") or {}, "resultado": res})
            respostas.append({"functionResponse": {"name": c["name"], "response": res}})
        contents.append({"role": "user", "parts": respostas})
    raise LLMError("Agente excedeu o limite de passos.")


def rodar(pergunta: str, conn: sqlite3.Connection, embedder: Embedder | None = None) -> dict:
    """Executa o agente: OpenAI/Gemini se houver chave, senão roteador offline."""
    f = Ferramentas(conn, embedder)
    provedor = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provedor == "openai" and os.getenv("OPENAI_API_KEY"):
        return _rodar_openai(
            pergunta, f, os.environ["OPENAI_API_KEY"], os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
    if os.getenv("GEMINI_API_KEY"):
        return _rodar_gemini(
            pergunta, f, os.environ["GEMINI_API_KEY"], os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        )
    if os.getenv("OPENAI_API_KEY"):
        return _rodar_openai(
            pergunta, f, os.environ["OPENAI_API_KEY"], os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
    return _rodar_offline(pergunta, f)
