from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from pypdf import PdfReader


def read_file(path: str, max_chars: int = 20000) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    try:
        if ext in (".txt", ".md"):
            text = p.read_text(encoding="utf-8", errors="replace")
        elif ext == ".csv":
            df = pd.read_csv(p)
            text = df.to_csv(index=False)
        elif ext in (".xlsx", ".xls"):
            dfs = pd.read_excel(p, sheet_name=None)
            text = "\n\n".join(
                f"--- Sheet: {name} ---\n{df.to_csv(index=False)}"
                for name, df in dfs.items()
            )
        elif ext == ".pdf":
            reader = PdfReader(str(p))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        elif ext == ".docx":
            doc = Document(str(p))
            text = "\n".join(para.text for para in doc.paragraphs)
        else:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = f"[binary file, {p.stat().st_size} bytes]"
    except Exception as e:
        return f"ERROR reading {path}: {e}"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text
