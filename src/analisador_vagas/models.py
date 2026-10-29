"""Modelos de dados e cálculo de match."""

from __future__ import annotations

from dataclasses import dataclass, field

SENIORIDADES = ("estagio", "junior", "pleno", "senior", "especialista", "nao_informado")


def normalizar_skill(skill: str) -> str:
    return " ".join(skill.strip().lower().split())


@dataclass
class AnaliseVaga:
    titulo: str
    senioridade: str
    skills_obrigatorias: list[str] = field(default_factory=list)
    skills_desejaveis: list[str] = field(default_factory=list)
    match_score: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> AnaliseVaga:
        sen = normalizar_skill(str(data.get("senioridade", "nao_informado")))
        if sen not in SENIORIDADES:
            sen = "nao_informado"
        return cls(
            titulo=str(data.get("titulo", "")).strip() or "Sem título",
            senioridade=sen,
            skills_obrigatorias=_limpar(data.get("skills_obrigatorias", [])),
            skills_desejaveis=_limpar(data.get("skills_desejaveis", [])),
        )

    def to_dict(self) -> dict:
        return {
            "titulo": self.titulo,
            "senioridade": self.senioridade,
            "skills_obrigatorias": self.skills_obrigatorias,
            "skills_desejaveis": self.skills_desejaveis,
            "match_score": self.match_score,
        }


def _limpar(skills: object) -> list[str]:
    if not isinstance(skills, list):
        return []
    vistos: list[str] = []
    for s in skills:
        n = normalizar_skill(str(s))
        if n and n not in vistos:
            vistos.append(n)
    return vistos


def calcular_match(analise: AnaliseVaga, minhas_skills: list[str]) -> float:
    """Score 0-100: obrigatórias pesam 0.8 e desejáveis 0.2 (determinístico)."""
    meus = {normalizar_skill(s) for s in minhas_skills if s.strip()}
    obr, des = analise.skills_obrigatorias, analise.skills_desejaveis
    if not obr and not des:
        return 0.0
    if not obr:
        return round(100 * sum(s in meus for s in des) / len(des), 1)
    p_obr = sum(s in meus for s in obr) / len(obr)
    if not des:
        return round(100 * p_obr, 1)
    p_des = sum(s in meus for s in des) / len(des)
    return round(100 * (0.8 * p_obr + 0.2 * p_des), 1)
