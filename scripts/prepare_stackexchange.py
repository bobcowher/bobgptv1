#!/usr/bin/env python3
"""Build question-and-answer pretraining sources from the Stack Exchange data dump.

    python scripts/prepare_stackexchange.py unix.stackexchange.com askubuntu.com
    python scripts/prepare_stackexchange.py stackoverflow.com --min-score 3

Reads data/raw/stackexchange/<site>.7z (from https://archive.org/details/stackexchange,
the April 2024 dump; Stack Overflow's posts are in stackoverflow.com-Posts.7z),
extracts Posts.xml and writes one record per question to
data/sources/se_<short>/<site>.jsonl, ready for scripts/build_mix.py.

A record is the question's title, tags and body followed by its best answers
(accepted first, then by score), with the HTML turned back into markdown-ish
text: code blocks become ``` fences, inline code `backticks`, list items "- ".
Questions that are closed, below --min-score, or have no answer that passes
the filter are dropped. Stack Overflow keeps only questions with a tag in
STACKOVERFLOW_TAGS, matching the code sources the mix already has.

Content is CC BY-SA 4.0; each record's metadata keeps the question URL, which
links to the authors.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import py7zr

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "stackexchange"
SOURCES_DIR = ROOT / "data" / "sources"

SHORT_NAMES = {
    "unix.stackexchange.com": "unix",
    "askubuntu.com": "askubuntu",
    "serverfault.com": "serverfault",
    "superuser.com": "superuser",
    "codereview.stackexchange.com": "codereview",
    "softwareengineering.stackexchange.com": "softeng",
    "stackoverflow.com": "stackoverflow",
}
STACKOVERFLOW_TAGS = {
    "python", "python-3.x", "c++", "rust", "bash", "shell", "sh", "zsh", "linux", "unix",
    "git", "sed", "awk", "grep", "command-line", "ubuntu", "makefile", "cmake", "ssh", "vim",
}
BLOCK_TAGS = {"p", "div", "ul", "ol", "blockquote", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "hr"}


class _Markdown(HTMLParser):
    """Post-body HTML to plain text, keeping code blocks and list structure."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.pre = 0

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.pre += 1
            self.out.append("\n\n```\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "br":
            self.out.append("\n")
        elif tag in BLOCK_TAGS:
            self.out.append("\n\n")
            if tag[0] == "h" and tag[1:].isdigit():
                self.out.append("#" * int(tag[1:]) + " ")
        elif tag == "img" and dict(attrs).get("alt"):
            self.out.append(f"[image: {dict(attrs)['alt']}]")

    def handle_endtag(self, tag):
        if tag == "pre":
            self.pre -= 1
            if not self.out[-1].endswith("\n"):
                self.out.append("\n")
            self.out.append("```\n\n")
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag in BLOCK_TAGS:
            self.out.append("\n\n")

    def handle_data(self, data):
        # HTML collapses whitespace everywhere except inside <pre>.
        self.out.append(data if self.pre else re.sub(r"\s+", " ", data))

    def text(self) -> str:
        # Tidy the gaps block breaks leave behind, but never inside a code fence.
        parts = "".join(self.out).split("```")
        for i in range(0, len(parts), 2):
            part = re.sub(r"[ \t]*\n[ \t]*", "\n", parts[i])
            parts[i] = re.sub(r"\n{3,}", "\n\n", part)
        return "```".join(parts).strip()


def html_to_text(html: str) -> str:
    parser = _Markdown()
    parser.feed(html)
    parser.close()
    return parser.text()


def parse_tags(raw: str) -> list[str]:
    # The 2024 dump writes "|python|pandas|"; older dumps "<python><pandas>".
    return [t for t in re.split(r"[<>|]", raw or "") if t]


def iter_rows(xml_path: Path):
    context = ET.iterparse(xml_path, events=("start", "end"))
    _, root = next(context)
    for event, element in context:
        if event == "end" and element.tag == "row":
            yield element.attrib
            root.clear()


def extract_posts(site: str) -> Path:
    archive = RAW_DIR / (f"{site}-Posts.7z" if site == "stackoverflow.com" else f"{site}.7z")
    out_dir = RAW_DIR / site
    xml_path = out_dir / "Posts.xml"
    if not xml_path.exists():
        print(f"{site}: extracting Posts.xml from {archive.name}")
        with py7zr.SevenZipFile(archive) as z:
            z.extract(path=out_dir, targets=["Posts.xml"])
    return xml_path


def collect(xml_path: Path, site: str, min_score: int, max_answers: int) -> dict[int, dict]:
    """Questions that pass the filters, each with its answers, in one pass (answers follow their question)."""
    questions: dict[int, dict] = {}
    for row in iter_rows(xml_path):
        post_type = row.get("PostTypeId")
        if post_type == "1":
            if row.get("ClosedDate") or int(row.get("Score", 0)) < min_score:
                continue
            tags = parse_tags(row.get("Tags", ""))
            if site == "stackoverflow.com" and not STACKOVERFLOW_TAGS.intersection(tags):
                continue
            questions[int(row["Id"])] = {
                "title": row.get("Title", ""),
                "body": row.get("Body", ""),
                "tags": tags,
                "score": int(row.get("Score", 0)),
                "accepted": int(row["AcceptedAnswerId"]) if row.get("AcceptedAnswerId") else None,
                "answers": [],
            }
        elif post_type == "2":
            question = questions.get(int(row.get("ParentId", -1)))
            if question is None:
                continue
            answer_id, score = int(row["Id"]), int(row.get("Score", 0))
            accepted = answer_id == question["accepted"]
            if score < 1 and not accepted:
                continue
            question["answers"].append((accepted, score, row.get("Body", "")))
            # Keep only the best few as we go so Stack Overflow fits in memory.
            if len(question["answers"]) > max_answers:
                question["answers"].sort(key=lambda a: (a[0], a[1]), reverse=True)
                question["answers"].pop()
    return questions


def render(question: dict) -> str:
    parts = [f"# {question['title']}"]
    if question["tags"]:
        parts.append("Tags: " + ", ".join(question["tags"]))
    parts.append(html_to_text(question["body"]))
    for accepted, score, body in sorted(question["answers"], key=lambda a: (a[0], a[1]), reverse=True):
        label = "Accepted answer" if accepted else "Answer"
        parts.append(f"## {label} (score {score})\n\n{html_to_text(body)}")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sites", nargs="+", choices=sorted(SHORT_NAMES))
    parser.add_argument("--min-score", type=int, default=1, help="drop questions scored below this")
    parser.add_argument("--max-answers", type=int, default=3, help="answers kept per question")
    args = parser.parse_args()

    for site in args.sites:
        questions = collect(extract_posts(site), site, args.min_score, args.max_answers)
        out_dir = SOURCES_DIR / f"se_{SHORT_NAMES[site]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{site}.jsonl"
        docs = chars = 0
        with out_path.open("w", encoding="utf-8") as f:
            for qid, question in questions.items():
                if not question["answers"]:
                    continue
                text = render(question)
                record = {
                    "id": f"stackexchange/{site}/{qid}",
                    "text": text,
                    "metadata": {
                        "url": f"https://{site}/q/{qid}",
                        "score": question["score"],
                        "tags": question["tags"],
                        "license": "CC-BY-SA-4.0",
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                docs += 1
                chars += len(text)
        print(f"{site}: {docs:,} questions, {chars:,} chars (~{chars // 4:,} tokens) -> {out_path}")


if __name__ == "__main__":
    main()
