import pandas as pd
from docx import Document

from gaia_agent.tools.files import read_file


def test_read_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world")
    assert "hello world" in read_file(str(p))


def test_read_csv(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("col1,col2\n1,2\n3,4\n")
    out = read_file(str(p))
    assert "col1" in out
    assert "3" in out


def test_read_xlsx(tmp_path):
    p = tmp_path / "a.xlsx"
    pd.DataFrame({"name": ["alice"], "age": [30]}).to_excel(p, index=False)
    out = read_file(str(p))
    assert "alice" in out
    assert "30" in out


def test_read_docx(tmp_path):
    p = tmp_path / "a.docx"
    doc = Document()
    doc.add_paragraph("Hello docx.")
    doc.save(p)
    assert "Hello docx." in read_file(str(p))


def test_read_unknown_falls_back(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"raw")
    out = read_file(str(p))
    assert "raw" in out or "binary" in out.lower()
