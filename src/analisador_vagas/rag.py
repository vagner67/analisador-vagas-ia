"""RAG sobre as vagas analisadas no SQLite.

Recuperação: embeddings de API (vetores cacheados na tabela `vaga_embeddings`) ou,
sem chave, TF-IDF offline. Geração: LLM com instrução de citar `[#id]`; sem LLM, resposta
extrativa (lista as vagas recuperadas). Toda resposta traz `fontes` (ids das vagas) e `modo`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field

from .embeddings import Embedder, TfidfIndex, cosseno

MODO_OFFLINE = "offline: TF-IDF lexical + resposta extrativa (sem LLM)"


@dataclass
class Resultado:
    resposta: str
    fontes: list[int]
    trechos: list[dict] = field(default_factory=list)
    modo: str = MODO_OFFLINE


def _documentos(conn: sqlite3.Connection) -> list[dict]:
    vagas = [
        dict(r) for r in conn.execute("SELECT id, titulo, senioridade, texto_original FROM vagas")
    ]
    skills: dict[int, list[str]] = {}
    for r in conn.execute("SELECT vaga_id, skill FROM vaga_skills ORDER BY tipo DESC, skill"):
        skills.setdefault(r["vaga_id"], []).append(r["skill"])
    for v in vagas:
        v["skills"] = skills.get(v["id"], [])
        v["doc"] = (
            f"{v['titulo']}\n{v['senioridade']}\n{' '.join(v['skills'])}\n{v['texto_original']}"
        )
    return vagas


def _vetores_api(conn: sqlite3.Connection, vagas: list[dict], emb: Embedder) -> list[list[float]]:
    cache = {
        r["vaga_id"]: json.loads(r["vetor"])
        for r in conn.execute(
            "SELECT vaga_id, vetor FROM vaga_embeddings WHERE modelo = ?", (emb.nome,)
        )
    }
    faltam = [v for v in vagas if v["id"] not in cache]
    if faltam:
        novos = emb.embed([v["doc"] for v in faltam])
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO vaga_embeddings (vaga_id, modelo, vetor) VALUES (?, ?, ?)",
                [
                    (v["id"], emb.nome, json.dumps(vet))
                    for v, vet in zip(faltam, novos, strict=True)
                ],
            )
        cache.update({v["id"]: vet for v, vet in zip(faltam, novos, strict=True)})
    return [cache[v["id"]] for v in vagas]


def buscar(
    conn: sqlite3.Connection, consulta: str, k: int = 3, embedder: Embedder | None = None
) -> tuple[list[dict], str]:
    """Retorna (hits, modo_recuperacao). Cada hit: id, titulo, senioridade, skills, score."""
    vagas = _documentos(conn)
    if not vagas or not consulta.strip():
        return [], embedder.nome if embedder else "tfidf-offline"
    if embedder is None:
        scores, modo = TfidfIndex([v["doc"] for v in vagas]).consultar(consulta), "tfidf-offline"
    else:
        vets = _vetores_api(conn, vagas, embedder)
        q = embedder.embed([consulta])[0]
        scores, modo = [cosseno(q, v) for v in vets], embedder.nome
    ordem = sorted(range(len(vagas)), key=lambda i: (-scores[i], vagas[i]["id"]))[:k]
    hits = [
        {
            "id": vagas[i]["id"],
            "titulo": vagas[i]["titulo"],
            "senioridade": vagas[i]["senioridade"],
            "skills": vagas[i]["skills"],
            "score": round(scores[i], 4),
            "trecho": vagas[i]["texto_original"][:400],
        }
        for i in ordem
        if scores[i] > 0
    ]
    return hits, modo


PROMPT_RAG = (
    "Responda à pergunta usando SOMENTE as vagas abaixo. Cite as vagas usadas no formato "
    "[#id]. Se a resposta não estiver nas vagas, diga que não sabe.\n\n{contexto}\n\n"
    "PERGUNTA: {pergunta}"
)


def perguntar(
    conn: sqlite3.Connection,
    pergunta: str,
    k: int = 3,
    embedder: Embedder | None = None,
    llm=None,
) -> Resultado:
    """`llm` precisa de `.responder(prompt) -> str` (GeminiClient/OpenAIClient) ou None."""
    hits, modo_rec = buscar(conn, pergunta, k, embedder)
    if not hits:
        return Resultado(
            "Nenhuma vaga relevante encontrada no banco.", [], [], f"recuperação: {modo_rec}"
        )
    ids = [h["id"] for h in hits]
    if llm is None:
        linhas = [
            f"[#{h['id']}] {h['titulo']} ({h['senioridade']}) — "
            f"skills: {', '.join(h['skills']) or '-'}"
            for h in hits
        ]
        texto = "Vagas mais relevantes (resposta extrativa, sem LLM):\n" + "\n".join(linhas)
        modo = MODO_OFFLINE if embedder is None else f"recuperação {modo_rec} + resposta extrativa"
        return Resultado(texto, ids, hits, modo)
    contexto = "\n\n".join(f"[#{h['id']}] {h['titulo']}\n{h['trecho']}" for h in hits)
    texto = llm.responder(PROMPT_RAG.format(contexto=contexto, pergunta=pergunta))
    citadas = [int(x) for x in re.findall(r"\[#(\d+)\]", texto)]
    fontes = [i for i in dict.fromkeys(citadas) if i in ids] or ids
    return Resultado(texto, fontes, hits, f"recuperação {modo_rec} + geração {llm.nome}")


def componentes_do_ambiente():
    """(embedder, llm) a partir do ambiente; ambos None => modo offline."""
    from .embeddings import embedder_do_ambiente
    from .llm import criar_cliente_do_ambiente

    emb = embedder_do_ambiente()
    return (emb, criar_cliente_do_ambiente()) if emb else (None, None)
