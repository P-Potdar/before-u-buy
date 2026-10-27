#!/usr/bin/env python3
"""Create and verify a clean source ZIP without local secrets or build output."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".vite", "dist", "dist-ssr", "coverage", "playwright-report",
    "test-results", "deliverables", "release",
}
EXCLUDED_FILES = {
    ".env", ".DS_Store", "App.js", "main.js",
}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if path.name in EXCLUDED_FILES:
        return False
    if path.suffix in {".pyc", ".zip", ".log"} or path.name.endswith(".tsbuildinfo"):
        return False
    if path.name.startswith(".env.") and path.name != ".env.example":
        return False
    return path.is_file() and not path.is_symlink()


def build_archive(destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in ROOT.rglob("*") if included(path))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    return len(files)


def verify_archive(path: Path) -> int:
    forbidden_parts = EXCLUDED_DIRS | {"__MACOSX"}
    required = {
        "README.md",
        ".env.example",
        "backend/assessment.py",
        "backend/main.py",
        "frontend/src/App.tsx",
        "frontend/src/main.tsx",
        "data/common/thirty_day_rule.md",
        "sentiment_labels.csv",
        "response_evaluation.csv",
        "TEST_RESULTS.md",
    }
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        corrupt = archive.testzip()
    if corrupt is not None:
        raise ValueError(f"ZIP contains a corrupt member: {corrupt}")
    missing = required - names
    if missing:
        raise ValueError(f"ZIP is missing required files: {sorted(missing)}")
    invalid = [
        name for name in names
        if any(part in forbidden_parts for part in Path(name).parts)
        or name == ".env"
        or name.endswith(".pyc")
        or name.endswith(".zip")
        or name.endswith(".tsbuildinfo")
        or name in {"frontend/src/App.js", "frontend/src/main.js"}
    ]
    if invalid:
        raise ValueError(f"ZIP contains excluded files: {invalid[:10]}")
    return len(names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=ROOT / "deliverables" / "BeforeYouBuy-AI.zip",
    )
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination == ROOT or destination == Path.home() or destination.name in {"", ".", ".."}:
        raise ValueError("Choose a specific ZIP file path for the deliverable")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_secrets.py")], check=True)
    count = build_archive(destination)
    verified = verify_archive(destination)
    print(f"Created and verified {destination} ({verified} files; {count} added)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Packaging failed: {error}", file=sys.stderr)
        raise SystemExit(1)
