CREATE TABLE IF NOT EXISTS vagas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    titulo TEXT NOT NULL,
    senioridade TEXT NOT NULL,
    match_score REAL NOT NULL,
    texto_original TEXT NOT NULL,
    provedor TEXT NOT NULL,
    criado_em TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS vaga_skills (
    vaga_id INTEGER NOT NULL REFERENCES vagas(id) ON DELETE CASCADE,
    skill TEXT NOT NULL,
    tipo TEXT NOT NULL CHECK (tipo IN ('obrigatoria', 'desejavel')),
    PRIMARY KEY (vaga_id, skill, tipo)
);

CREATE INDEX IF NOT EXISTS idx_vaga_skills_skill ON vaga_skills(skill);
