from gaia_agent.tools.python_exec import run_python


def test_run_python_captures_stdout():
    out = run_python("print(2 + 2)")
    assert "4" in out


def test_run_python_captures_last_expr():
    out = run_python("x = 10\ny = 32\nx + y")
    assert "42" in out


def test_run_python_reports_exceptions():
    out = run_python("1/0")
    assert "ZeroDivisionError" in out


def test_run_python_timeout():
    out = run_python("while True: pass", timeout=2)
    assert "timed out" in out.lower()
