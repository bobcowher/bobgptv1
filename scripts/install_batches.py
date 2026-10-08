#!/usr/bin/env python3
"""Copy validated Codex batches from a staging directory into a source.

    python scripts/install_batches.py /data/datasets/bobgptv1-staging/chat chat
    python scripts/install_batches.py /data/datasets/bobgptv1-staging/tutor_qa tutor_qa

Every staged batch file is re-validated (with the source's MAX_TOKENS) and
skipped if it fails. Records whose user turns repeat an earlier record's (across
all batches, in file order) are dropped, since validate.py only checks within
the files it is given. Writes data/sources/<source>/<slug>.jsonl.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

MAX_TOKENS = {"chat": "400", "tutor_qa": "400", "tutor_scripts": "400"}  # tutor_qa round 3+ (1024 context) allows 400


def main() -> None:
    staging, source = Path(sys.argv[1]), sys.argv[2]
    out_dir = Path(__file__).resolve().parent.parent / "data" / "sources" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MAX_TOKENS=MAX_TOKENS.get(source, "230"))

    seen, files, kept, dropped = set(), 0, 0, 0
    for path in sorted((staging / "batches").glob("*.jsonl")):
        check = subprocess.run([sys.executable, str(staging / "validate.py"), str(path)],
                               capture_output=True, text=True, env=env)
        if check.returncode:
            print(f"skip {path.name}: {check.stdout.strip().splitlines()[-1]}")
            continue
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            question = " ".join(m["content"] for m in record["messages"] if m["role"] == "user").strip().lower()
            if question in seen:
                dropped += 1
                continue
            seen.add(question)
            records.append(line)
        (out_dir / path.name).write_text("\n".join(records) + "\n", encoding="utf-8")
        files += 1
        kept += len(records)
    print(f"{source}: {files} batches, {kept} records installed, {dropped} duplicates dropped")


if __name__ == "__main__":
    main()
