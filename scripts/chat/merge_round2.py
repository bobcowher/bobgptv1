#!/usr/bin/env python3
"""One-off: append batches_round2.json themes to batches.json (english / general)."""

import json
from pathlib import Path

HERE = Path(__file__).absolute().parent
batches = json.loads((HERE / "batches.json").read_text())
known = {b["slug"] for b in batches}
for b in json.loads((HERE / "batches_round2.json").read_text()):
    if b["slug"] not in known:
        batches.append({"slug": b["slug"], "language": "english", "topic": b["topic"],
                        "level": "general", "guidance": b["guidance"]})
(HERE / "batches.json").write_text(json.dumps(batches, indent=1) + "\n")
print(len(batches), "batches")
