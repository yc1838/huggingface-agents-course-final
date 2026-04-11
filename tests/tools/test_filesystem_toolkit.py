import os
import shutil
import tempfile
from pathlib import Path
import pytest
from gaia_agent.tools.files import read_file, ls, grep, glob_files, write_file

class TestFilesystemToolkit:
    @pytest.fixture(autouse=True)
    def setup_sandbox(self):
        self.test_dir = tempfile.mkdtemp()
        self.sandbox = Path(self.test_dir)
        yield
        shutil.rmtree(self.test_dir)

    def test_write_and_read_file(self):
        file_path = str(self.sandbox / "test.txt")
        content = "Hello, world!\nLine 2\nLine 3"
        
        # Test write
        result = write_file(file_path, content)
        assert "Successfully wrote" in result
        assert (self.sandbox / "test.txt").exists()
        
        # Test read (existing functionality enhanced)
        read_content = read_file(file_path)
        assert "Hello, world!" in read_content
        
    def test_ls_directory(self):
        (self.sandbox / "file1.txt").touch()
        (self.sandbox / "subdir").mkdir()
        (self.sandbox / "subdir" / "file2.txt").touch()
        
        # Test ls root
        result = ls(str(self.sandbox))
        assert "file1.txt" in result
        assert "subdir/" in result
        
    def test_grep_content(self):
        file_path = str(self.sandbox / "grep_test.txt")
        content = "Apple\nBanana\nCherry\nApple Pie"
        write_file(file_path, content)
        
        # Test grep
        result = grep("Apple", file_path)
        assert "Line 1: Apple" in result
        assert "Line 4: Apple Pie" in result
        assert "Banana" not in result
        
    def test_glob_files(self):
        (self.sandbox / "a.py").touch()
        (self.sandbox / "b.txt").touch()
        (self.sandbox / "c.py").touch()
        
        # We need to set the search context or pass absolute paths
        # Assuming glob_files works relative to a base or takes patterns
        result = glob_files(str(self.sandbox / "*.py"))
        assert "a.py" in result
        assert "c.py" in result
        assert "b.txt" not in result

    def test_read_file_with_chunking(self):
        file_path = str(self.sandbox / "long.txt")
        content = "\n".join([f"Line {i}" for i in range(1, 11)])
        write_file(file_path, content)
        
        # Test chunking (lines 2 to 4)
        # Note: API might change to support start/end lines
        result = read_file(file_path, start_line=2, end_line=4)
        assert "Line 2" in result
        assert "Line 4" in result
        assert "Line 1" not in result
        assert "Line 5" not in result

    def test_error_handling_missing_file(self):
        result = read_file(str(self.sandbox / "nonexistent.txt"))
        assert "ERROR" in result

    def test_error_handling_directory_as_file(self):
        dir_path = str(self.sandbox / "my_dir")
        os.mkdir(dir_path)
        result = read_file(dir_path)
        assert "ERROR" in result
