"""Interface de linha de comando."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

from . import db
from .llm import LLMClient, LLMError, criar_cliente_do_ambiente
from .service import analisar


def _skills(valor: str | None) -> list[str]:
    valor = valor if valor is not None else os.getenv("MINHAS_SKILLS", "")
    return [s for s in valor.split(",") if s.strip()]


def main(argv: list[str] | None = None, cliente: LLMClient | None = None) -> int:
    p = argparse.ArgumentParser(prog="analisador-vagas", description="Analisa vagas com LLM.")
    p.add_argument("--db", default=os.getenv("VAGAS_DB", "vagas.db"), help="arquivo SQLite")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analisar", help="analisa uma vaga (arquivo ou '-' para stdin)")
    a.add_argument("arquivo", nargs="?", default="-")
    a.add_argument("--skills", help="suas skills separadas por vírgula (ou MINHAS_SKILLS)")
    a.add_argument("--nao-salvar", action="store_true")
    r = sub.add_parser("relatorio", help="relatórios SQL")
    r.add_argument("tipo", choices=["top-skills", "senioridade", "vagas"])
    r.add_argument("--limite", type=int, default=10)
    r.add_argument("--so", choices=["obrigatoria", "desejavel"], help="filtra tipo de skill")
    q = sub.add_parser("perguntar", help="RAG: pergunta sobre as vagas salvas (cita ids)")
    q.add_argument("pergunta")
    q.add_argument("-k", type=int, default=3)
    g = sub.add_parser("agente", help="agente com tool-calling sobre as vagas salvas")
    g.add_argument("pergunta")
    args = p.parse_args(argv)

    conn = db.conectar(args.db)
    try:
        if args.cmd == "analisar":
            texto = (
                sys.stdin.read() if args.arquivo == "-" else Path(args.arquivo).read_text("utf-8")
            )
            try:
                cli = cliente or criar_cliente_do_ambiente()
                res = analisar(texto, cli, _skills(args.skills))
            except (LLMError, ValueError) as e:
                print(f"Erro: {e}", file=sys.stderr)
                return 1
            saida = res.to_dict()
            if not args.nao_salvar:
                saida["id"] = db.salvar(conn, res, texto, cli.nome)
            print(json.dumps(saida, ensure_ascii=False, indent=2))
        elif args.cmd == "perguntar":
            from . import rag

            try:
                emb, llm_ = rag.componentes_do_ambiente()
                res = rag.perguntar(conn, args.pergunta, args.k, emb, llm_)
            except (LLMError, httpx.HTTPError) as e:
                print(f"Erro: {e}", file=sys.stderr)
                return 1
            print(json.dumps(res.__dict__, ensure_ascii=False, indent=2))
        elif args.cmd == "agente":
            from . import agente
            from .embeddings import embedder_do_ambiente

            try:
                saida = agente.rodar(args.pergunta, conn, embedder_do_ambiente())
            except (LLMError, httpx.HTTPError) as e:
                print(f"Erro: {e}", file=sys.stderr)
                return 1
            print(json.dumps(saida, ensure_ascii=False, indent=2))
        else:
            if args.tipo == "top-skills":
                linhas = db.top_skills(conn, args.limite, args.so)
            elif args.tipo == "senioridade":
                linhas = db.por_senioridade(conn)
            else:
                linhas = db.listar(conn, args.limite)
            print(json.dumps(linhas, ensure_ascii=False, indent=2))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
