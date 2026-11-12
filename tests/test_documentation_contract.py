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
        *Path(ROOT / "speechprefix/stages").rglob("*.md"),
        *Path(ROOT / "speechprefix/pipelines").rglob("*.md"),
        ROOT / "README.md",
        ROOT / "CONTEXT.md",
        ROOT / "AGENTS.md",
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
    assert "seven days before the event" in spec
    assert "frozen comparison baseline" in spec
    assert "seven days before" in summary
    assert "seven-day window" in context
    assert "story-graph research track" in context
    description = metadata["project"]["description"].lower()
    assert "pre-event news context" in description
    assert "continuation" not in description


def test_speech_content_research_layers_have_separate_paths() -> None:
    research_layers = ROOT / "research/idea_layers"
    docs_layers = ROOT / "docs/idea-layer"
    for layer in ("outcomes", "points", "topics", "relations", "stories", "prediction", "legacy_atoms"):
        assert (research_layers / layer).is_dir()
    for layer in ("transcript-outcomes", "points", "topics", "relations", "stories", "prediction", "evaluation", "sources", "governance"):
        assert (docs_layers / layer).is_dir()
    assert (research_layers / "README.md").is_file()
    for layer in ("outcomes", "points", "topics", "relations", "stories", "prediction", "legacy_atoms"):
        assert (ROOT / "tests/idea_layers" / layer).is_dir()
    assert not (ROOT / "research/group_speech_ideas_into_topics.py").exists()
    assert not (ROOT / "research/extract_layered_transcript_outcomes.py").exists()
    layout = (research_layers / "README.md").read_text(encoding="utf-8").lower()
    assert "never same-event features" in layout
    assert "data/source-archives/speech-content/<same-layer>/" in layout


def test_future_stage_scaffolds_exist_without_claiming_implementation() -> None:
    ideas = (ROOT / "speechprefix/stages/ideas/README.md").read_text(encoding="utf-8").lower()
    sentences = (ROOT / "speechprefix/stages/sentences/README.md").read_text(encoding="utf-8").lower()
    pipelines = (ROOT / "speechprefix/pipelines/README.md").read_text(encoding="utf-8").lower()
    assert "active package" in ideas
    assert "model.py" in ideas
    assert "no sentence model" in sentences
    assert "no end-to-end pipeline is implemented" in pipelines


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
    assert "context-idea-predict" in output
    assert "legacy speaker-history phrase baseline" in output
    assert "legacy speaker-history phrase forecast" in output
    assert "legacy speaker-history baseline evaluation" in output
