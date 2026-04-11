from gaia_agent.tools.python_exec import run_python

def test_run_python_shows_line_numbers_on_error():
    code = """
import math
x = 10
y = 0
# This should fail
z = x / y
"""
    result = run_python(code.strip())
    assert "Line 5: z = x / y" in result
    assert "ZeroDivisionError" in result

def test_run_python_success_no_line_numbers():
    code = "print('Hello')"
    result = run_python(code)
    assert "Hello" in result
    assert "Line 1:" not in result
