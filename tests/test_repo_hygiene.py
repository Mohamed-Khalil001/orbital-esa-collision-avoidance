"""Guards against committing things GitHub rejects or that do not belong in the repository."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}
LIMIT_BYTES = 50 * 1024 * 1024  # GitHub warns above 50 MB and rejects above 100 MB


def _files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not (set(path.relative_to(ROOT).parts) & SKIP_DIRS):
            yield path


def test_no_file_is_close_to_the_github_size_limit():
    too_big = [str(p.relative_to(ROOT)) for p in _files() if p.stat().st_size > LIMIT_BYTES]
    assert not too_big, f"files over 50 MB: {too_big}"


def test_raw_competition_data_is_git_ignored():
    """Local copies of the big ESA files are fine on a developer machine as long as git ignores them."""
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/*.csv", "data/*.zip", "data/raw/"):
        assert pattern in ignored, f"{pattern} must be listed in .gitignore"
    assert "!data/sample/" in ignored
