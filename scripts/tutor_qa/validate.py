#!/usr/bin/env python3
"""Validate tutor Q&A batch files.

    python validate.py batches/python_closures_intro.jsonl [...]

Checks every record for: schema, id prefix, unique ids/questions, token budget
(the model's context is 256 GPT-2 tokens), and that code in the *answer*
compiles: ```python via ast.parse, ```rust via rustc, ```cpp via g++. Code in
the *question* may be intentionally broken (debugging exercises) and is not
checked. Exit status 1 on any failure.
"""

import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import tiktoken

MAX_TOKENS = 230
ROLES = {"system", "user", "assistant"}
FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
enc = tiktoken.get_encoding("gpt2")
CACHE = Path(__file__).with_name(".compile_cache")
CACHE.mkdir(exist_ok=True)


def render(messages):
    headers = {"system": "### System", "user": "### Question", "assistant": "### Answer"}
    return "\n\n".join(f"{headers[m['role']]}\n{m['content'].strip()}" for m in messages) + "\n\n### End"


def compile_ok(lang, code):
    key = hashlib.sha256(f"{lang}\0{code}".encode()).hexdigest()
    if (CACHE / key).exists():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        if lang == "rust":
            src = Path(tmp) / "snippet.rs"
            src.write_text("#![allow(dead_code, unused)]\n" + code)
            crate = "bin" if re.search(r"\bfn\s+main\s*\(", code) else "lib"
            cmd = ["rustc", "--edition", "2021", "--crate-type", crate, "--emit=metadata",
                   "-o", str(Path(tmp) / "out"), str(src)]
        else:
            src = Path(tmp) / "snippet.cpp"
            src.write_text(code)
            cmd = ["g++", "-std=c++20", "-fsyntax-only", "-w", str(src)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode == 0:
        (CACHE / key).touch()
        return None
    return result.stderr.strip().splitlines()[0] if result.stderr.strip() else "compile failed"


def check_file(path, seen_ids, seen_questions):
    errors = []
    prefix = f"tutor_qa/{Path(path).stem}_"
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        where = f"{path}:{lineno}"
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{where}: invalid JSON: {e}")
            continue
        rid = rec.get("id", "")
        if not rid.startswith(prefix):
            errors.append(f"{where}: id {rid!r} must start with {prefix!r}")
        if rid in seen_ids:
            errors.append(f"{where}: duplicate id {rid!r}")
        seen_ids.add(rid)
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            errors.append(f"{where}: messages must be a list of >= 2 turns")
            continue
        if any(set(m) != {"role", "content"} or m["role"] not in ROLES or not m["content"].strip() for m in msgs):
            errors.append(f"{where}: each message needs exactly role/content, non-empty")
            continue
        turns = [m["role"] for m in msgs if m["role"] != "system"]
        if turns[0] != "user" or turns[-1] != "assistant" or any(a == b for a, b in zip(turns, turns[1:])):
            errors.append(f"{where}: turns must alternate user/assistant, starting user, ending assistant")
        if msgs[0]["role"] == "system" and any(m["role"] == "system" for m in msgs[1:]):
            errors.append(f"{where}: system turn only allowed first")
        meta = rec.get("metadata", {})
        for key in ("topic", "language", "level", "kind"):
            if not meta.get(key):
                errors.append(f"{where}: metadata.{key} missing")
        n = len(enc.encode_ordinary(render(msgs)))
        if n > MAX_TOKENS:
            errors.append(f"{where}: {n} tokens rendered (max {MAX_TOKENS})")
        q = " ".join(m["content"] for m in msgs if m["role"] == "user").strip().lower()
        if q in seen_questions:
            errors.append(f"{where}: duplicate question")
        seen_questions.add(q)
        for m in msgs:
            if m["role"] != "assistant":
                continue
            for lang, code in FENCE.findall(m["content"]):
                lang = lang.lower()
                if lang in ("python", "py"):
                    try:
                        ast.parse(code)
                    except SyntaxError as e:
                        errors.append(f"{where}: answer python does not parse: {e}")
                elif lang in ("rust", "rs", "cpp", "c++"):
                    err = compile_ok("rust" if lang in ("rust", "rs") else "cpp", code)
                    if err:
                        errors.append(f"{where}: answer {lang} does not compile: {err}")
    return errors


def main():
    seen_ids, seen_questions, errors, count = set(), set(), [], 0
    for path in sys.argv[1:]:
        errors += check_file(path, seen_ids, seen_questions)
        count += sum(1 for _ in open(path, encoding="utf-8"))
    for e in errors:
        print(e)
    print(f"{count} records, {len(errors)} errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
