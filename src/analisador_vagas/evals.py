"""Evals offline: extração de skills (precisão/recall) e recuperação (hit@k, MRR).

Dataset: tests/evals/*.jsonl (vagas escritas à mão, rotuladas manualmente).
Roda sem chave de API: avalia o modo offline (palavras-chave + TF-IDF).
Uso: python -m analisador_vagas.evals [--dir tests/evals] [--markdown]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path

from . import db, rag
from .heuristica import HeuristicaClient
from .models import AnaliseVaga


def _ler(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]


def avaliar_extracao(vagas: list[dict], cliente=None) -> dict:
    cliente = cliente or HeuristicaClient()
    tp = fp = fn = 0
    for v in vagas:
        a = AnaliseVaga.from_dict(cliente.extrair(v["texto"]))
        pred = set(a.skills_obrigatorias) | set(a.skills_desejaveis)
        ouro = set(v["skills_obrigatorias"]) | set(v["skills_desejaveis"])
        tp += len(pred & ouro)
        fp += len(pred - ouro)
        fn += len(ouro - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precisao": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "n": len(vagas)}


def montar_banco(vagas: list[dict], conn: sqlite3.Connection, cliente=None) -> None:
    cliente = cliente or HeuristicaClient()
    for v in vagas:
        a = AnaliseVaga.from_dict(cliente.extrair(v["texto"]))
        vid = db.salvar(conn, a, v["texto"], cliente.nome)
        assert vid == v["id"], "ids do dataset devem ser sequenciais a partir de 1"


def avaliar_recuperacao(consultas: list[dict], conn: sqlite3.Connection, ks=(1, 3), emb=None):
    res: dict = {}
    rr = 0.0
    for k in ks:
        acertos = 0
        for c in consultas:
            hits, _ = rag.buscar(conn, c["consulta"], k, emb)
            acertos += any(h["id"] in c["relevantes"] for h in hits)
        res[f"hit@{k}"] = acertos / len(consultas)
    for c in consultas:
        hits, _ = rag.buscar(conn, c["consulta"], 10, emb)
        pos = next((i for i, h in enumerate(hits, 1) if h["id"] in c["relevantes"]), None)
        rr += 1 / pos if pos else 0.0
    res["mrr"] = rr / len(consultas)
    res["n"] = len(consultas)
    return res


def rodar(diretorio: str | Path = "tests/evals") -> dict:
    d = Path(diretorio)
    vagas = _ler(d / "vagas_rotuladas.jsonl")
    consultas = _ler(d / "retrieval.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        conn = db.conectar(Path(tmp) / "eval.db")
        try:
            montar_banco(vagas, conn)
            rec = avaliar_recuperacao(consultas, conn)
        finally:
            conn.close()
    return {"extracao": avaliar_extracao(vagas), "recuperacao": rec}


def markdown(r: dict) -> str:
    e, q = r["extracao"], r["recuperacao"]
    return "\n".join(
        [
            "| Eval (offline mode) | Metric | Value |",
            "|---|---|---|",
            f"| Skill extraction (n={e['n']} postings) | precision | {e['precisao']:.3f} |",
            f"| | recall | {e['recall']:.3f} |",
            f"| | F1 | {e['f1']:.3f} |",
            f"| Retrieval (TF-IDF, n={q['n']} queries) | hit@1 | {q['hit@1']:.3f} |",
            f"| | hit@3 | {q['hit@3']:.3f} |",
            f"| | MRR | {q['mrr']:.3f} |",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m analisador_vagas.evals")
    p.add_argument("--dir", default="tests/evals")
    p.add_argument("--markdown", action="store_true")
    a = p.parse_args(argv)
    r = rodar(a.dir)
    print(markdown(r) if a.markdown else json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
