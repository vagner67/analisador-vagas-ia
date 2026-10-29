"""Clientes LLM (Gemini e OpenAI) com saída JSON estruturada via REST."""

from __future__ import annotations

import json
import os
from typing import Protocol

import httpx

PROMPT = (
    "Você é um analisador de vagas de emprego. Extraia da vaga abaixo: "
    "titulo, senioridade (um de: estagio, junior, pleno, senior, especialista, "
    "nao_informado), skills_obrigatorias e skills_desejaveis (listas de nomes "
    "curtos de tecnologias/competências, em minúsculas). Responda apenas JSON.\n\n"
    "VAGA:\n"
)

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "titulo": {"type": "string"},
        "senioridade": {
            "type": "string",
            "enum": ["estagio", "junior", "pleno", "senior", "especialista", "nao_informado"],
        },
        "skills_obrigatorias": {"type": "array", "items": {"type": "string"}},
        "skills_desejaveis": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["titulo", "senioridade", "skills_obrigatorias", "skills_desejaveis"],
}


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    nome: str

    def extrair(self, texto_vaga: str) -> dict: ...


class GeminiClient:
    nome = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", timeout: float = 60) -> None:
        self.api_key, self.model, self.timeout = api_key, model, timeout

    def extrair(self, texto_vaga: str) -> dict:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        schema = {k: v for k, v in SCHEMA.items()}
        body = {
            "contents": [{"parts": [{"text": PROMPT + texto_vaga}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "temperature": 0,
            },
        }
        r = httpx.post(
            url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=self.timeout
        )
        if r.status_code != 200:
            raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
        try:
            texto = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(texto)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMError(f"Resposta inválida do Gemini: {e}") from e


class OpenAIClient:
    nome = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout: float = 60) -> None:
        self.api_key, self.model, self.timeout = api_key, model, timeout

    def extrair(self, texto_vaga: str) -> dict:
        schema = dict(SCHEMA, additionalProperties=False)
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT + texto_vaga}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "analise_vaga", "strict": True, "schema": schema},
            },
        }
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise LLMError(f"OpenAI HTTP {r.status_code}: {r.text[:300]}")
        try:
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMError(f"Resposta inválida da OpenAI: {e}") from e


def criar_cliente_do_ambiente() -> LLMClient:
    """Cria o cliente a partir de LLM_PROVIDER (gemini|openai) e da chave correspondente."""
    provedor = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provedor == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("Defina GEMINI_API_KEY no ambiente.")
        return GeminiClient(key, os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    if provedor == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("Defina OPENAI_API_KEY no ambiente.")
        return OpenAIClient(key, os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    raise LLMError(f"LLM_PROVIDER inválido: {provedor!r} (use 'gemini' ou 'openai').")
