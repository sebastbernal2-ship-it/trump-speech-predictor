from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _markdown_files() -> list[Path]:
    return [
        *Path(ROOT / "docs").rglob("*.md"),
        ROOT / "README.md",
        ROOT / "CONTEXT.md",
        ROOT / "data/README.md",
    ]


def test_local_markdown_links_resolve() -> None:
    failures = []
    for path in _markdown_files():
        for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            destination = (path.parent / target.split("#", 1)[0]).resolve()
            if not destination.exists():
                failures.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not failures, "Broken local Markdown links:\n" + "\n".join(failures)


def test_canonical_task_and_package_description_stay_aligned() -> None:
    spec = (ROOT / "docs/idea-layer/README.md").read_text(encoding="utf-8").lower()
    summary = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8").lower()
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "seven days before that event" in spec
    assert "seven days before" in summary
    assert "seven days before" in context
    description = metadata["project"]["description"].lower()
    assert "preceding-week news context" in description
    assert "continuation" not in description


def test_cli_marks_history_forecasters_as_legacy() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "speechprefix", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = re.sub(r"\s+", " ", re.sub(r"(?<=-)\n\s*", "", result.stdout))
    assert "active idea-prediction workflow is context-idea-classifier-evaluate" in output
    assert "legacy speaker-history phrase baseline" in output
    assert "legacy speaker-history phrase forecast" in output
    assert "legacy speaker-history baseline evaluation" in output
