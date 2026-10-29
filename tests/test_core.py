import json

import httpx
import pytest

from analisador_vagas import db, llm
from analisador_vagas.cli import main
from analisador_vagas.models import AnaliseVaga, calcular_match
from analisador_vagas.service import analisar


def test_from_dict_normaliza(resposta_padrao):
    a = AnaliseVaga.from_dict(resposta_padrao)
    assert a.senioridade == "junior"
    assert a.skills_obrigatorias == ["python", "sql"]


def test_senioridade_invalida():
    assert AnaliseVaga.from_dict({"senioridade": "xyz"}).senioridade == "nao_informado"


def test_match(resposta_padrao):
    a = AnaliseVaga.from_dict(resposta_padrao)
    assert calcular_match(a, ["Python", "sql", "fastapi"]) == 90.0
    assert calcular_match(a, []) == 0.0
    assert calcular_match(AnaliseVaga("x", "junior"), ["python"]) == 0.0


def test_analisar_vazio(fake):
    with pytest.raises(ValueError):
        analisar("   ", fake, [])


def test_db_e_relatorios(fake, tmp_path):
    conn = db.conectar(tmp_path / "t.db")
    for skills in (["python"], ["python", "sql"]):
        a = analisar("vaga", fake, skills)
        db.salvar(conn, a, "vaga", "fake")
    top = db.top_skills(conn)
    assert {"skill": "python", "vagas": 2} in top
    assert db.top_skills(conn, limite=1, tipo="obrigatoria") == [{"skill": "python", "vagas": 2}]
    assert db.top_skills(conn, tipo="desejavel")[0]["skill"] == "docker"
    assert db.por_senioridade(conn)[0]["vagas"] == 2
    assert db.listar(conn)[0]["match_score"] == 80.0


def test_sql_injection_seguro(tmp_path):
    conn = db.conectar(tmp_path / "t.db")
    a = AnaliseVaga("'); DROP TABLE vagas; --", "junior", ["x'y"])
    db.salvar(conn, a, "t", "fake")
    assert db.listar(conn)[0]["titulo"].startswith("'); DROP")


def test_cli(fake, tmp_path, capsys):
    f = tmp_path / "vaga.txt"
    f.write_text("Vaga Python", encoding="utf-8")
    dbp = str(tmp_path / "c.db")
    assert main(["--db", dbp, "analisar", str(f), "--skills", "python,sql"], cliente=fake) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["match_score"] == 80.0 and out["id"] == 1
    assert main(["--db", dbp, "relatorio", "top-skills"]) == 0
    assert "python" in capsys.readouterr().out


def test_cliente_ambiente(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(llm.LLMError):
        llm.criar_cliente_do_ambiente()
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert llm.criar_cliente_do_ambiente().nome == "openai"
    monkeypatch.setenv("LLM_PROVIDER", "outro")
    with pytest.raises(llm.LLMError):
        llm.criar_cliente_do_ambiente()


def test_gemini_parse(monkeypatch, resposta_padrao):
    def post(url, **kw):
        assert kw["headers"]["x-goog-api-key"] == "k"
        body = {"candidates": [{"content": {"parts": [{"text": json.dumps(resposta_padrao)}]}}]}
        return httpx.Response(200, json=body)

    monkeypatch.setattr(llm.httpx, "post", post)
    assert llm.GeminiClient("k").extrair("v")["titulo"] == resposta_padrao["titulo"]


def test_openai_erro(monkeypatch):
    monkeypatch.setattr(llm.httpx, "post", lambda url, **kw: httpx.Response(401, text="no"))
    with pytest.raises(llm.LLMError):
        llm.OpenAIClient("k").extrair("v")


def test_api(fake, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from analisador_vagas import api

    monkeypatch.setenv("VAGAS_DB", str(tmp_path / "a.db"))
    api.app.dependency_overrides[api.get_cliente] = lambda: fake
    c = TestClient(api.app)
    r = c.post("/analisar", json={"texto": "vaga", "skills": ["python"]})
    assert r.status_code == 200 and r.json()["id"] == 1
    assert "python" in [r["skill"] for r in c.get("/relatorios/top-skills").json()]
    api.app.dependency_overrides.clear()


def test_heuristica_extrai():
    from analisador_vagas.heuristica import HeuristicaClient

    texto = (
        "Desenvolvedor Python Sênior\nRequisitos: Python, SQL e Docker\nDesejável: AWS e FastAPI\n"
    )
    a = analisar(texto, HeuristicaClient(), ["python", "sql", "aws"])
    assert a.titulo == "Desenvolvedor Python Sênior"
    assert a.senioridade == "senior"
    assert a.skills_obrigatorias == ["python", "sql", "docker"]
    assert a.skills_desejaveis == ["fastapi", "aws"] or set(a.skills_desejaveis) == {
        "aws",
        "fastapi",
    }
    assert a.match_score == round(100 * (0.8 * 2 / 3 + 0.2 * 1 / 2), 1)


def test_api_sem_chave_usa_modo_sem_llm(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from analisador_vagas import api

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("VAGAS_DB", str(tmp_path / "h.db"))
    c = TestClient(api.app)
    assert "Analisador de Vagas" in c.get("/").text
    assert c.get("/health").json()["modo"] == "modo sem LLM"
    r = c.post("/analisar", json={"texto": "Dev Júnior\nRequisitos: Python", "salvar": False})
    d = r.json()
    assert r.status_code == 200 and d["usa_llm"] is False and "sem LLM" in d["modo"]
    assert d["skills_obrigatorias"] == ["python"] and d["senioridade"] == "junior"
