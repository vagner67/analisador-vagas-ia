"""Evals offline em pytest: falham se as métricas regredirem abaixo do piso."""

from pathlib import Path

from analisador_vagas import evals

DIR = Path(__file__).parent / "evals"


def test_evals_offline_pisos():
    r = evals.rodar(DIR)
    e, q = r["extracao"], r["recuperacao"]
    assert e["n"] == 12 and q["n"] == 14
    assert e["precisao"] >= 0.85 and e["recall"] >= 0.75
    assert q["hit@3"] >= 0.85 and q["mrr"] >= 0.8
