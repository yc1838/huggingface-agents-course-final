"""GAIA agent tools. Registry assembled by build_tools()."""
from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from gaia_agent.config import Config
from gaia_agent.tools.audio import transcribe_audio as _transcribe_audio
from gaia_agent.tools.files import read_file as _read_file
from gaia_agent.tools.pdf import inspect_pdf as _inspect_pdf
from gaia_agent.tools.python_exec import run_python as _run_python
from gaia_agent.tools.search import tavily_search as _tavily_search
from gaia_agent.tools.vision import inspect_visual_content as _inspect_visual_content
from gaia_agent.tools.web import fetch_url as _fetch_url
from gaia_agent.tools.youtube import youtube_transcript as _youtube_transcript


def build_tools(cfg: Config) -> list[BaseTool]:
    """Build the list of LangChain tools, injecting config via closures."""

    @tool
    def tavily_search(query: str, max_results: int = 5) -> str:
        """Search the web via Tavily. Returns a list of title/url/snippet results."""
        return _tavily_search(query, api_key=cfg.tavily_api_key, max_results=max_results)

    @tool
    def fetch_url(url: str, max_chars: int = 8000) -> str:
        """Fetch a URL and return its extracted main text (trafilatura)."""
        return _fetch_url(url, max_chars=max_chars)

    @tool
    def run_python(code: str, timeout: int = 30) -> str:
        """Run Python code in a sandboxed subprocess and return stdout + last expression."""
        return _run_python(code, timeout=timeout)

    @tool
    def read_file(path: str, max_chars: int = 20000) -> str:
        """Read a local file (txt, md, csv, xlsx, pdf, docx) and return its text."""
        return _read_file(path, max_chars=max_chars)

    @tool
    def transcribe_audio(path: str) -> str:
        """Transcribe an audio file using faster-whisper. Returns the transcript text."""
        return _transcribe_audio(path, model_size=cfg.whisper_model)

    @tool
    def inspect_pdf(url_or_path: str, query: str) -> str:
        """Extract text from a PDF (local or remote). Recommended for reading academic papers or long documents."""
        return _inspect_pdf(url_or_path, query=query)

    @tool
    def inspect_visual_content(file_path_or_url: str, prompt: str) -> str:
        """Use Gemini Multimodal Vision to analyze an image (PNG, JPG) or video (MP4). Use this for counting objects on screen or identifying visual details."""
        return _inspect_visual_content(file_path_or_url, prompt=prompt)

    @tool
    def youtube_transcript(url: str) -> str:
        """Fetch the transcript for a YouTube video URL."""
        return _youtube_transcript(url)

    return [
        tavily_search,
        fetch_url,
        run_python,
        read_file,
        transcribe_audio,
        youtube_transcript,
        inspect_pdf,
        inspect_visual_content,
    ]
