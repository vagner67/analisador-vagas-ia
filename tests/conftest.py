import pytest


class FakeLLM:
    """Cliente LLM falso, usado SOMENTE nos testes."""

    nome = "fake"

    def __init__(self, resposta: dict) -> None:
        self.resposta = resposta
        self.chamadas: list[str] = []

    def extrair(self, texto_vaga: str) -> dict:
        self.chamadas.append(texto_vaga)
        return self.resposta


@pytest.fixture
def resposta_padrao() -> dict:
    return {
        "titulo": "Desenvolvedor Python Júnior",
        "senioridade": "Junior",
        "skills_obrigatorias": ["Python", "SQL", "python"],
        "skills_desejaveis": ["FastAPI", "Docker"],
    }


@pytest.fixture
def fake(resposta_padrao: dict) -> FakeLLM:
    return FakeLLM(resposta_padrao)
