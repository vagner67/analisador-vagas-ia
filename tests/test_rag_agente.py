import json
from pathlib import Path

import httpx
import pytest

from analisador_vagas import agente, db, embeddings, evals, rag

EVALS = Path(__file__).parent / "evals"


@pytest.fixture
def conn(tmp_path, monkeypatch):
    for v in ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(v, raising=False)
    c = db.conectar(tmp_path / "r.db")
    evals.montar_banco(evals._ler(EVALS / "vagas_rotuladas.jsonl"), c)
    yield c
    c.close()


def test_tfidf_rank():
    idx = embeddings.TfidfIndex(["python fastapi", "java spring", "react typescript"])
    s = idx.consultar("vaga de java")
    assert s.index(max(s)) == 1


def test_perguntar_offline_cita_fontes(conn):
    r = rag.perguntar(conn, "vagas com spark e airflow")
    assert r.fontes[0] == 2 and "[#2]" in r.resposta and "sem LLM" in r.modo


def test_perguntar_banco_vazio(tmp_path):
    r = rag.perguntar(db.conectar(tmp_path / "v.db"), "python")
    assert r.fontes == []


class FakeEmb:
    nome = "fake-emb"

    def __init__(self):
        self.chamadas = 0

    def embed(self, textos):
        self.chamadas += 1
        return [[1.0 if "spark" in t.lower() else 0.0, 1.0] for t in textos]


class FakeLLM:
    nome = "fake-llm"

    def responder(self, prompt):
        assert "[#2]" in prompt
        return "A vaga [#2] pede Spark. [#999] não existe."


def test_perguntar_com_embeddings_e_llm_cacheia_e_filtra_citacoes(conn):
    emb = FakeEmb()
    r = rag.perguntar(conn, "spark", 3, emb, FakeLLM())
    assert r.fontes == [2] and "fake-llm" in r.modo
    n = conn.execute("SELECT COUNT(*) FROM vaga_embeddings").fetchone()[0]
    assert n == 12
    rag.buscar(conn, "spark", 3, emb)
    assert emb.chamadas == 3  # 1 lote de docs + 2 consultas (docs vêm do cache)


def test_openai_embedder(monkeypatch):
    def post(url, **kw):
        assert url.endswith("/embeddings")
        return httpx.Response(
            200, json={"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
        )

    monkeypatch.setattr(embeddings.httpx, "post", post)
    assert embeddings.OpenAIEmbedder("k").embed(["a", "b"]) == [[1.0], [2.0]]


def test_gemini_embedder(monkeypatch):
    def post(url, **kw):
        assert "batchEmbedContents" in url and len(kw["json"]["requests"]) == 1
        return httpx.Response(200, json={"embeddings": [{"values": [0.5]}]})

    monkeypatch.setattr(embeddings.httpx, "post", post)
    assert embeddings.GeminiEmbedder("k").embed(["a"]) == [[0.5]]


def test_router_offline():
    assert agente.rotear_offline("quais as skills mais pedidas?")[0][0] == "gerar_relatorio_sql"
    assert agente.rotear_offline("vagas por senioridade")[0][1] == {"tipo": "senioridade"}
    plano = agente.rotear_offline("qual o match da vaga 3 se eu sei: react, typescript")
    assert plano == [("calcular_match", {"vaga_id": 3, "skills": ["react", "typescript"]})]
    assert agente.rotear_offline("vagas de java")[0][0] == "buscar_vagas"


def test_agente_offline_match(conn):
    out = agente.rodar("vagas de dados compatíveis, tenho: python, sql, spark", conn)
    assert "regex" in out["modo"]
    nomes = [p["ferramenta"] for p in out["passos"]]
    assert nomes[0] == "buscar_vagas" and "calcular_match" in nomes
    m = next(p for p in out["passos"] if p["args"].get("vaga_id") == 2)
    assert m["resultado"]["match_score"] == pytest.approx(100 * 0.8 * 3 / 5, abs=0.1)
    assert "airflow" in m["resultado"]["faltando_obrigatorias"]


def test_ferramenta_invalida(conn):
    f = agente.Ferramentas(conn)
    assert "erro" in f.executar("rm_rf", {})
    assert "erro" in f.executar("calcular_match", {"vaga_id": 999, "skills": []})


def test_agente_openai_tool_calling(conn, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    respostas = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "gerar_relatorio_sql",
                        "arguments": json.dumps({"tipo": "top_skills", "limite": 1}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Python é a mais pedida."},
    ]
    enviados = []

    def post(url, **kw):
        enviados.append(json.loads(json.dumps(kw["json"])))
        return httpx.Response(200, json={"choices": [{"message": respostas[len(enviados) - 1]}]})

    monkeypatch.setattr(agente.httpx, "post", post)
    out = agente.rodar("skill mais pedida?", conn)
    assert out["resposta"] == "Python é a mais pedida." and out["modo"].startswith("openai")
    assert out["passos"][0]["resultado"]["linhas"][0]["skill"] == "python"
    assert enviados[1]["messages"][-1]["role"] == "tool"
    assert enviados[0]["tools"][0]["function"]["name"] == "buscar_vagas"


def test_agente_gemini_tool_calling(conn, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    partes = [
        [
            {
                "functionCall": {
                    "name": "calcular_match",
                    "args": {"vaga_id": 4, "skills": ["power bi", "excel", "sql"]},
                }
            }
        ],
        [{"text": "Match de 80 na vaga [#4]."}],
    ]
    n = {"i": 0}

    def post(url, **kw):
        p = partes[n["i"]]
        n["i"] += 1
        return httpx.Response(
            200, json={"candidates": [{"content": {"role": "model", "parts": p}}]}
        )

    monkeypatch.setattr(agente.httpx, "post", post)
    # embeddings não são usados nesta ferramenta; passamos None
    out = agente.rodar("match vaga 4", conn, None)
    assert out["passos"][0]["resultado"]["match_score"] == 80.0
    assert "[#4]" in out["resposta"] and out["modo"].startswith("gemini")


def test_api_perguntar_e_agente(conn, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from analisador_vagas import api

    dbp = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("VAGAS_DB", dbp)
    c = TestClient(api.app)
    r = c.post("/perguntar", json={"pergunta": "quem pede power bi?"}).json()
    assert r["fontes"][0] == 4
    a = c.post("/agente", json={"pergunta": "vagas por senioridade"}).json()
    assert a["passos"][0]["ferramenta"] == "gerar_relatorio_sql"


def test_cli_perguntar(conn, capsys):
    from analisador_vagas.cli import main

    dbp = conn.execute("PRAGMA database_list").fetchone()["file"]
    assert main(["--db", dbp, "perguntar", "backend java spring kafka", "-k", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["fontes"] == [6]
    assert main(["--db", dbp, "agente", "skills mais pedidas"]) == 0
    assert "python" in capsys.readouterr().out


def test_api_demo_seed(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from analisador_vagas import api

    for v in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("DEMO_SEED", "1")
    monkeypatch.setenv("VAGAS_DB", str(tmp_path / "seed.db"))
    r = TestClient(api.app).post("/perguntar", json={"pergunta": "kubernetes terraform"}).json()
    assert r["fontes"] and "sem LLM" in r["modo"]
