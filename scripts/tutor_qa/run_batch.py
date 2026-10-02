#!/usr/bin/env python3
"""Run one Codex batch headlessly: python run_batch.py <slug>. Skips batches that already validate."""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent  # not resolve(): chat/ symlinks this file
PYTHON = sys.executable
LANG_NAMES = {"python": "Python 3", "rust": "Rust (2021 edition)", "cpp": "C++20", "linux": "Linux/bash"}
LEVEL_HINTS = {
    "beginner": "someone in their first months of programming: plain words, small examples, common mistakes.",
    "intermediate": "someone comfortable with the basics: idioms, edge cases, trade-offs, real-world bugs.",
}

slug = sys.argv[1]
batch = next(b for b in json.loads((HERE / "batches.json").read_text()) if b["slug"] == slug)
out = HERE / "batches" / f"{slug}.jsonl"
validate = [PYTHON, str(HERE / "validate.py"), str(out)]

if out.exists() and subprocess.run(validate, capture_output=True).returncode == 0:
    print(f"{slug}: already valid, skipping")
    sys.exit(0)

# A batch with a neutral original in batches_neutral/ is restyled, not written from scratch.
template = "restyle_template.md" if (HERE / "batches_neutral" / f"{slug}.jsonl").exists() else "prompt_template.md"
voice = HERE / "voice.md"
prompt = (HERE / template).read_text().format(
    language_name=LANG_NAMES.get(batch["language"], batch["language"]),
    level_hint=LEVEL_HINTS.get(batch["level"], ""),
    voice=voice.read_text().strip() if voice.exists() else "",
    python=PYTHON, **batch)
log = HERE / "logs" / f"{slug}.log"
with log.open("w") as f:
    subprocess.run(
        ["codex", "exec", "-C", str(HERE), "--skip-git-repo-check", "--sandbox", "workspace-write",
         "--ephemeral", "-o", str(HERE / "logs" / f"{slug}.last"), prompt],
        stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

result = subprocess.run(validate, capture_output=True, text=True)
print(f"{slug}: {result.stdout.strip().splitlines()[-1] if result.stdout.strip() else 'no output'}")
