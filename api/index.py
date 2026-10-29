"""Entrada Vercel (runtime Python): expõe o app FastAPI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analisador_vagas.api import app  # noqa: E402

__all__ = ["app"]
