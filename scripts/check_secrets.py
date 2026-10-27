#!/usr/bin/env python3
"""Fail if source files contain secret-shaped values. Never prints the match."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "dist-ssr",
    ".pytest_cache",
    "__pycache__",
    "coverage",
    "playwright-report",
    "test-results",
    "release",
    ".vite",
}
RULES = {
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "assigned_secret": re.compile(
        r"(?i)(?:api_key|secret|token|password)\s*=\s*['\"][^'\"]{8,}['\"]"
    ),
}
PLACEHOLDERS = ("changeme", "example", "placeholder", "your-", "dummy", "fake", "xxx", "todo")


def allowed(rule: str, text: str, match: re.Match[str]) -> bool:
    if rule != "assigned_secret":
        return False
    snippet = match.group(0).lower()
    return any(token in snippet for token in PLACEHOLDERS)


def main() -> int:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in {".pyc", ".png", ".jpg", ".zip", ".node", ".woff", ".woff2"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pattern in RULES.items():
            for match in pattern.finditer(text):
                if allowed(name, text, match):
                    continue
                findings.append(f"{path.relative_to(ROOT)} matched {name}")
                break
    if findings:
        print("Secret scan failed. Matches are not printed.")
        print("\n".join(findings))
        return 1
    print("Secret scan passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
