"""Modo sem LLM: extração offline por palavras-chave (sem IA, determinística).

Usado quando nenhuma chave GEMINI_API_KEY/OPENAI_API_KEY está configurada, para que a
demo funcione sem segredos. NÃO é saída de LLM: identifica apenas termos de um
vocabulário fixo e classifica-os como obrigatórios/desejáveis pelo contexto da linha.
"""

from __future__ import annotations

import re
import unicodedata

SKILLS: dict[str, list[str]] = {
    "python": [r"python"],
    "java": [r"java(?!\s*script)"],
    "javascript": [r"javascript", r"\bjs\b"],
    "typescript": [r"typescript", r"\bts\b"],
    "go": [r"\bgolang\b", r"\bgo\b(?=\s*(lang|,|/|\)|$))"],
    "c#": [r"c#", r"\.net\b"],
    "php": [r"\bphp\b"],
    "ruby": [r"\bruby\b"],
    "rust": [r"\brust\b"],
    "kotlin": [r"\bkotlin\b"],
    "sql": [r"\bsql\b"],
    "postgresql": [r"postgres(ql)?"],
    "mysql": [r"\bmysql\b"],
    "mongodb": [r"mongo(db)?"],
    "redis": [r"\bredis\b"],
    "fastapi": [r"fastapi"],
    "django": [r"\bdjango\b"],
    "flask": [r"\bflask\b"],
    "spring": [r"\bspring\b"],
    "node.js": [r"node(\.?js)?\b"],
    "react": [r"\breact\b"],
    "angular": [r"\bangular\b"],
    "vue": [r"\bvue(\.?js)?\b"],
    "docker": [r"\bdocker\b"],
    "kubernetes": [r"kubernetes", r"\bk8s\b"],
    "aws": [r"\baws\b", r"amazon web services"],
    "azure": [r"\bazure\b"],
    "gcp": [r"\bgcp\b", r"google cloud"],
    "terraform": [r"terraform"],
    "git": [r"\bgit\b", r"github", r"gitlab"],
    "ci/cd": [r"ci\s*/\s*cd", r"github actions", r"jenkins"],
    "linux": [r"\blinux\b"],
    "rest": [r"\brest(ful)?\b", r"\bapis?\b"],
    "graphql": [r"graphql"],
    "pandas": [r"\bpandas\b"],
    "machine learning": [r"machine learning", r"aprendizado de maquina"],
    "llm": [r"\bllms?\b", r"genai", r"ia generativa", r"openai", r"gemini"],
    "power bi": [r"power\s*bi"],
    "excel": [r"\bexcel\b"],
    "spark": [r"\bspark\b"],
    "airflow": [r"airflow"],
    "testes automatizados": [r"pytest", r"testes? (automatizados|unitarios)", r"\btdd\b"],
    "scrum": [r"\bscrum\b", r"\bageis?\b", r"\bagil\b", r"agile"],
    "ingles": [r"ingles", r"english"],
}

DESEJAVEL = re.compile(
    r"desejave|diferencia|nice to have|plus\b|bonus|seria um|preferencia|preferred"
)
OBRIGATORIO = re.compile(r"requisito|obrigatori|necessari|required|must|exigid|imprescind")

SENIORIDADE = [
    ("especialista", r"especialista|staff|principal|\blead\b|tech lead"),
    ("senior", r"senior|\bsr\b"),
    ("pleno", r"pleno|\bpl\b|mid[- ]level"),
    ("junior", r"junior|\bjr\b|entry[- ]level"),
    ("estagio", r"estagi|intern(ship)?\b|trainee"),
]


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def extrair_por_palavras_chave(texto: str) -> dict:
    linhas = [ln.strip() for ln in texto.splitlines() if ln.strip()]
    obr: list[str] = []
    des: list[str] = []
    secao = "obrigatoria"
    for ln in linhas:
        n = _norm(ln)
        if DESEJAVEL.search(n):
            secao = "desejavel"
        elif OBRIGATORIO.search(n):
            secao = "obrigatoria"
        for skill, padroes in SKILLS.items():
            if any(re.search(p, n) for p in padroes):
                alvo = des if secao == "desejavel" else obr
                if skill not in obr and skill not in des:
                    alvo.append(skill)
    todo = _norm(texto)
    senioridade = next((s for s, p in SENIORIDADE if re.search(p, todo)), "nao_informado")
    titulo = linhas[0][:120] if linhas else "Sem título"
    return {
        "titulo": titulo,
        "senioridade": senioridade,
        "skills_obrigatorias": obr,
        "skills_desejaveis": des,
    }


class HeuristicaClient:
    """Cliente 'modo sem LLM' — mesma interface de LLMClient, sem chamada externa."""

    nome = "modo sem LLM (palavras-chave)"

    def extrair(self, texto_vaga: str) -> dict:
        return extrair_por_palavras_chave(texto_vaga)
