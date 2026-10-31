# Job Posting Analyzer with AI — RAG · Tool-calling Agent · Evals

[![CI](https://github.com/vagner67/analisador-vagas-ia/actions/workflows/ci.yml/badge.svg)](https://github.com/vagner67/analisador-vagas-ia/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Live demo:** https://analisador-vagas-ia.vercel.app (runs in labelled *offline mode*: no API key is configured on the demo)

**English** · [Português](#português)

Python + FastAPI service that turns free-text job postings into structured data with an LLM
(**Gemini** or **OpenAI**, structured JSON output via `httpx`), stores them in **SQLite**, and adds:

- **RAG** (`perguntar`): ask questions about the stored postings; answers **cite posting ids** (`[#id]`).
  Embeddings from the Gemini (`gemini-embedding-001`) or OpenAI (`text-embedding-3-small`) API, vectors cached in SQLite
  (`vaga_embeddings`). Without a key: pure-Python **TF-IDF** retrieval + extractive answer, labelled `offline`.
- **Agent with function calling** (`agente`): OpenAI `tools` / Gemini `functionDeclarations` loop with three tools:
  `buscar_vagas` (RAG search), `calcular_match`, `gerar_relatorio_sql`. Every tool call is returned in `passos`.
  Without a key: a deterministic regex router, labelled `offline`.
- **Evals**: hand-labelled dataset in `tests/evals/*.jsonl` → skill-extraction precision/recall/F1 and retrieval hit@k/MRR,
  enforced as regression floors in pytest and published in every CI run summary.
- **Docker** + docker-compose; **Vercel** serverless deploy; ruff + pytest + GitHub Actions.

No heavy dependencies (no torch / numpy / vector DB): runtime deps are `httpx` and `fastapi`.

## Architecture

```mermaid
flowchart LR
    U[User / Web demo / CLI] --> API[FastAPI · api/index.py on Vercel]
    API -->|/analisar| EX{API key?}
    EX -->|yes| LLM[Gemini / OpenAI<br/>structured JSON]
    EX -->|no| KW[Keyword extractor<br/>offline]
    LLM --> DB[(SQLite<br/>vagas · vaga_skills · vaga_embeddings)]
    KW --> DB
    API -->|/perguntar| RAG[RAG retriever]
    RAG -->|key| EMB[Embeddings API<br/>vectors cached in SQLite]
    RAG -->|no key| TF[TF-IDF offline]
    RAG --> GEN[LLM answer with &#91;#id&#93; citations<br/>or extractive offline answer]
    API -->|/agente| AG[Agent loop<br/>function calling / offline router]
    AG --> T1[buscar_vagas] --> RAG
    AG --> T2[calcular_match] --> DB
    AG --> T3[gerar_relatorio_sql] --> DB
    EV[Evals · tests/evals/*.jsonl] -.-> KW
    EV -.-> TF
```

## Run

```bash
./scripts/dev.sh setup                # venv + pip install -e ".[dev,api]"
./scripts/dev.sh check                # ruff + pytest + offline evals (same as CI)
./scripts/dev.sh run                  # uvicorn with --reload on :8000 (PORT=... to change)
```

Manual equivalent:

```bash
pip install -e ".[dev]"
export GEMINI_API_KEY=...            # optional (or LLM_PROVIDER=openai + OPENAI_API_KEY)
analisador-vagas analisar exemplos/vaga_exemplo.txt --skills python,sql
analisador-vagas perguntar "Which postings require Kubernetes?"
analisador-vagas agente "Qual o match da vaga 2 se eu sei python, sql e spark?"
analisador-vagas relatorio top-skills
uvicorn analisador_vagas.api:app      # POST /analisar /analisar/lote /perguntar /agente · GET /relatorios/top-skills /health
python -m analisador_vagas.evals --markdown
docker compose up --build             # http://localhost:8000 (DEMO_SEED=1 loads example postings)
```

## Eval results (offline mode, actual output of `python -m analisador_vagas.evals --markdown`)

| Eval (offline mode) | Metric | Value |
|---|---|---|
| Skill extraction (n=12 postings) | precision | 0.921 |
| | recall | 0.795 |
| | F1 | 0.853 |
| Retrieval (TF-IDF, n=14 queries) | hit@1 | 0.929 |
| | hit@3 | 0.929 |
| | MRR | 0.929 |

Dataset: 12 **hand-written, illustrative** postings (not scraped real ads) labelled manually, and 14 retrieval queries
(incl. 2 English queries against Portuguese text — the lexical TF-IDF baseline misses one of them, which is why
hit@k < 1). Keyword extraction misses skills outside its fixed vocabulary (e.g. dbt, Kafka, LangChain), hence recall 0.79.
LLM-mode evals require an API key and are not part of CI.

Live check (manual run on 2026-10-29 with `GEMINI_API_KEY`, `gemini-embedding-001` + `gemini-3.5-flash`):
"Quais vagas pedem Kubernetes?" → answer citing `[#7]` and `[#5]` (the two postings that list Kubernetes);
the agent called `calcular_match(vaga_id=2, skills=[python, sql, spark])` and answered 48 with Airflow/AWS missing.

## Design decisions / RAG trade-offs

What the code actually does today, and what it costs:

- **No chunking — one document per posting.** Each posting is indexed as a single document
  (`title + seniority + extracted skills + original text`, see `rag._documentos`), truncated to 8,000 chars before
  embedding. Postings are short, and the question is usually "which posting…", so whole-document retrieval keeps
  citations trivially correct (`[#id]` = one posting). Trade-off: long postings lose their tail, and a question about
  one paragraph competes with the whole document's signal; the LLM context uses only the first 400 chars per hit.
- **Retrieval = brute-force cosine, no vector DB.** Vectors are cached as JSON in the SQLite table
  `vaga_embeddings`, keyed by `(vaga_id, model)`, and embedded lazily only for postings missing from the cache. Scoring
  is an O(n) Python loop. Fine for hundreds of postings and zero extra infra; it would need an ANN index (sqlite-vec,
  pgvector, FAISS) at scale. Switching embedding models never mixes vectors because the model name is part of the key.
- **Embeddings are optional.** With a key: `gemini-embedding-001` or `text-embedding-3-small` via plain REST
  (`httpx`, no SDK). Without a key: a pure-Python **TF-IDF** index (accent-folded tokens, PT/EN stopwords, sublinear
  tf). That is lexical, not semantic — the evals show it misses an English query against Portuguese text — and every
  response is labelled `modo: offline` so it never pretends to be an LLM.
- **Generation with enforced citations.** The prompt demands `[#id]`; cited ids are parsed and filtered to the
  retrieved set (hallucinated ids are dropped; if none are cited, all retrieved ids are returned). Offline, the answer
  is extractive (a list of the retrieved postings) instead of generated.
- **No re-ranking, no hybrid search, no query rewriting.** Kept out deliberately to keep the pipeline small and
  testable offline; hybrid BM25+vector would be the first addition if evals on real data showed misses.
- **Evals.** A hand-labelled set in `tests/evals/` measures skill-extraction precision/recall and retrieval
  hit@1/hit@3/MRR. They run offline in pytest and CI (published to the job summary). LLM-mode evals need a key and
  are not in CI — the offline numbers are a floor/baseline, not a measure of the LLM path.
- **Async where it pays off.** `POST /analisar/lote` is an `async def` endpoint that analyses up to 20 postings
  concurrently (`assincrono.analisar_lote`). The LLM clients use blocking `httpx`, so each call runs in a worker
  thread via `asyncio.to_thread`, bounded by an `asyncio.Semaphore` (`concorrencia`, 1–8) to respect provider rate
  limits, with optional per-item timeout and per-item error isolation. Trade-off: threads instead of a native
  `httpx.AsyncClient` avoid duplicating every client, at the cost of one thread per in-flight call. Tests
  (`tests/test_async.py`, anyio) assert real overlap, the semaphore cap, ordering, timeouts and error isolation.
- **SQLite** for storage and SQL reports: zero-ops, file-based; on Vercel it lives in `/tmp` and is ephemeral
  (demo is re-seeded with example postings).

---

# Português

**Demo ao vivo:** https://analisador-vagas-ia.vercel.app (modo offline rotulado, sem chave de API)

Serviço Python + FastAPI que transforma vagas em dados estruturados com LLM (Gemini ou OpenAI), salva em SQLite e oferece:

- **RAG** (`perguntar`): perguntas sobre as vagas salvas, com **citação dos ids** das vagas (`[#id]`). Embeddings via API
  (Gemini/OpenAI) com vetores cacheados no SQLite; sem chave, **TF-IDF** em Python puro + resposta extrativa (rotulado *offline*).
- **Agente com function calling** (`agente`): ferramentas `buscar_vagas`, `calcular_match`, `gerar_relatorio_sql`;
  sem chave, roteador determinístico por regex (rotulado). Cada chamada de ferramenta aparece em `passos`.
- **Evals**: dataset rotulado à mão em `tests/evals/` com precisão/recall de extração de skills e hit@k/MRR de recuperação,
  verificados no pytest e publicados no resumo de cada execução do CI (tabela acima, números reais).
- **Docker/docker-compose**, deploy **Vercel**, ruff, pytest e GitHub Actions.

Como rodar: veja a seção *Run* acima (os comandos são os mesmos). O match score é determinístico:
skills obrigatórias pesam 80% e desejáveis 20%. Sem chave de API, nenhuma resposta finge ser de LLM: o campo `modo` indica
`offline`.

Licença MIT.

## Decisões de design / trade-offs do RAG

- **Sem chunking — um documento por vaga** (`título + senioridade + skills + texto`), truncado em 8.000 caracteres
  para embedding. Citação fica trivialmente correta (`[#id]` = uma vaga); em troca, vagas longas perdem o final e o
  contexto do LLM usa só os 400 primeiros caracteres de cada vaga recuperada.
- **Recuperação por cosseno em força bruta, sem banco vetorial.** Vetores cacheados em JSON na tabela SQLite
  `vaga_embeddings` com chave `(vaga_id, modelo)` e gerados só para vagas novas. Serve para centenas de vagas sem
  infra extra; em escala precisaria de índice ANN (sqlite-vec, pgvector, FAISS).
- **Embeddings opcionais.** Com chave: `gemini-embedding-001` ou `text-embedding-3-small` via REST. Sem chave:
  **TF-IDF** em Python puro — busca lexical, não semântica (os evals mostram a falha numa pergunta em inglês sobre
  texto em português), sempre rotulada `modo: offline`.
- **Geração com citação obrigatória**: ids citados são validados contra os recuperados (ids inventados são
  descartados). Offline, a resposta é extrativa.
- **Sem re-ranking, busca híbrida ou reescrita de consulta** — de propósito, para manter o pipeline pequeno e
  testável offline; BM25+vetor seria o primeiro acréscimo.
- **Evals** rotulados à mão (precisão/recall de skills, hit@k/MRR) rodam offline no pytest e no CI; evals do modo
  LLM exigem chave e não estão no CI.
- **Async onde compensa**: `POST /analisar/lote` (`async def`) analisa até 20 vagas em paralelo com
  `asyncio.to_thread` + `asyncio.Semaphore` (limite 1–8, respeita rate limit), timeout e isolamento de erro por item.
  Threads em vez de `httpx.AsyncClient` evitam duplicar os clientes; testes em `tests/test_async.py` (anyio).
- **SQLite**: zero operação; na Vercel fica em `/tmp` (efêmero, demo repovoada com vagas de exemplo).

Atalhos: `./scripts/dev.sh setup | test | lint | evals | run | check`.
