"""Persistência em SQLite com SQL puro e consultas parametrizadas."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

from .models import AnaliseVaga


def conectar(caminho: str | Path = "vagas.db") -> sqlite3.Connection:
    conn = sqlite3.connect(str(caminho))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    schema = resources.files("analisador_vagas").joinpath("schema.sql").read_text("utf-8")
    conn.executescript(schema)
    return conn


def salvar(conn: sqlite3.Connection, a: AnaliseVaga, texto: str, provedor: str) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO vagas (titulo, senioridade, match_score, texto_original, provedor) "
            "VALUES (?, ?, ?, ?, ?)",
            (a.titulo, a.senioridade, a.match_score, texto, provedor),
        )
        vaga_id = int(cur.lastrowid or 0)
        linhas = [(vaga_id, s, "obrigatoria") for s in a.skills_obrigatorias]
        linhas += [(vaga_id, s, "desejavel") for s in a.skills_desejaveis]
        conn.executemany(
            "INSERT OR IGNORE INTO vaga_skills (vaga_id, skill, tipo) VALUES (?, ?, ?)", linhas
        )
    return vaga_id


def top_skills(conn: sqlite3.Connection, limite: int = 10, tipo: str | None = None) -> list[dict]:
    sql = (
        "SELECT skill, COUNT(DISTINCT vaga_id) AS vagas FROM vaga_skills "
        "WHERE (? IS NULL OR tipo = ?) GROUP BY skill ORDER BY vagas DESC, skill LIMIT ?"
    )
    return [dict(r) for r in conn.execute(sql, (tipo, tipo, limite))]


def por_senioridade(conn: sqlite3.Connection) -> list[dict]:
    sql = (
        "SELECT senioridade, COUNT(*) AS vagas, ROUND(AVG(match_score), 1) AS match_medio "
        "FROM vagas GROUP BY senioridade ORDER BY vagas DESC"
    )
    return [dict(r) for r in conn.execute(sql)]


def listar(conn: sqlite3.Connection, limite: int = 20) -> list[dict]:
    sql = (
        "SELECT id, titulo, senioridade, match_score, criado_em FROM vagas "
        "ORDER BY match_score DESC, id DESC LIMIT ?"
    )
    return [dict(r) for r in conn.execute(sql, (limite,))]
