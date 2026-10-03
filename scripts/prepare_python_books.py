#!/usr/bin/env python3
"""Build the python_books source: openly licensed Python books and courses.

Only sources we could share are included (permissive or share-alike; no
NonCommercial / NoDerivatives). Each is pinned to a commit; its license
statement is verified in the archive and kept in licenses/.

Output is data/sources/python_books/<name>.jsonl, one record per chapter/file,
ready for scripts/build_mix.py. Download/normalize helpers are shared with
prepare_python_corpus.py.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import re
import sys
import tarfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_python_corpus import fetch, normalize_text, relative_archive_path


@dataclass(frozen=True)
class Book:
    name: str
    repository: str
    revision: str
    license: str
    license_file: str        # file in the archive that states the license
    license_fragment: str    # text that must appear in it
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()

    @property
    def archive_url(self) -> str:
        return f"https://codeload.github.com/{self.repository}/tar.gz/{self.revision}"


BOOKS = (
    Book("practical_python", "dabeaz-course/practical-python", "93dca856b41c61a0a0f85ae334116e4c125629ea",
         "CC-BY-SA-4.0", "LICENSE.md", "Attribution-ShareAlike 4.0",
         ("Notes/*.md", "Solutions/*.py")),
    Book("python_mastery", "dabeaz-course/python-mastery", "a55856bf898579c4aafe7fe0e1c9f5466939cb30",
         "CC-BY-SA-4.0", "LICENSE.md", "Attribution-ShareAlike 4.0",
         ("Exercises/*.md", "Solutions/*.py")),
    Book("byte_of_python", "swaroopch/byte-of-python", "929abecc51f453a5ea1b4fb6aa4b4e815c150e33",
         "CC-BY-SA-4.0", "README.md", "Creative Commons Attribution-ShareAlike 4",
         ("*.md", "programs/*.py"),
         ("README.md", "SUMMARY.md", "INSTALL.md", "revision_history.md", "*/*.md")),
    Book("dive_into_python3", "diveintomark/diveintopython3", "793871b16676b3bb2536df6a5dc90b8404eb6f91",
         "CC-BY-SA-3.0", "about.html", "Creative Commons Attribution-ShareAlike 3",
         ("*.html", "examples/*.py"),
         ("*/*.html", "blank.html", "colophon.html")),
    Book("exercism_python", "exercism/python", "6105f6d9d3ef66ce3d5c8f529d2882af3d76b706",
         "MIT", "LICENSE", "Permission is hereby granted, free of charge",
         ("concepts/*/about.md", "concepts/*/introduction.md", "exercises/*/.docs/*.md",
          "exercises/*/.meta/exemplar.py", "exercises/*/.meta/example.py"),
         ("*_test.py",)),
    Book("learn_python", "trekhleb/learn-python", "5e4fad5903ce5a76cc1f90838d69a220968e1494",
         "MIT", "LICENSE", "Permission is hereby granted, free of charge",
         ("src/*.py",)),
    Book("the_algorithms_python", "TheAlgorithms/Python", "dcb8c64b4847ceaae81623c6d5ac11a68be5f7f9",
         "MIT", "LICENSE.md", "Permission is hereby granted, free of charge",
         ("*.py",),
         (".*", "scripts/*", "*/tests/*", "*/test_*.py", "*_test.py")),
)


class _HTMLText(HTMLParser):
    """Readable text from a book chapter: drop scripts/styles, keep <pre> verbatim."""
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "dt", "dd", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip, self.pre = [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag == "pre":
            self.pre += 1
            self.parts.append("\n\n")
        elif tag in self.BLOCK:
            self.parts.append("\n\n" if tag.startswith("h") or tag == "p" else "\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1
        elif tag == "pre":
            self.pre -= 1
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self.skip:
            return
        # Collapse whitespace runs but keep them: inline tags (<code>, <em>) split words otherwise.
        self.parts.append(data if self.pre else re.sub(r"\s+", " ", data))


def html_to_text(raw: bytes) -> bytes:
    parser = _HTMLText()
    parser.feed(raw.decode("utf-8", errors="replace"))
    lines = [line.rstrip() for line in "".join(parser.parts).splitlines()]
    lines = [line[1:] if line.startswith(" ") and not line.startswith("  ") else line for line in lines]
    text, blank = [], 0
    for line in lines:
        blank = blank + 1 if not line.strip() else 0
        if blank <= 1:
            text.append(line)
    return "\n".join(text).encode("utf-8")


def selected(path: str, book: Book) -> bool:
    return (any(fnmatch.fnmatchcase(path, p) for p in book.include)
            and not any(fnmatch.fnmatchcase(path, p) for p in book.exclude))


def build(output_dir: Path) -> None:
    (output_dir / "licenses").mkdir(parents=True, exist_ok=True)
    seen, provenance, total_docs, total_chars = set(), [], 0, 0
    for book in BOOKS:
        archive = fetch(book.archive_url)
        files = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member in tar:
                rel = relative_archive_path(member.name)
                if member.isfile() and rel is not None and member.size <= 1_000_000:
                    files[rel] = tar.extractfile(member).read()

        license_text = files.get(book.license_file, b"").decode("utf-8", errors="replace")
        if book.license_fragment.lower() not in license_text.lower():
            raise ValueError(f"{book.name}: license statement not found in {book.license_file}")
        (output_dir / "licenses" / f"{book.name}_{PurePosixPath(book.license_file).name}").write_bytes(
            files[book.license_file])

        records, rejected = [], 0
        for path in sorted(files):
            if not selected(path, book):
                continue
            raw = html_to_text(files[path]) if path.endswith(".html") else files[path]
            text = normalize_text(raw)
            if text is None:
                rejected += 1
                continue
            if path.endswith(".py"):
                try:
                    ast.parse(text)
                except SyntaxError:
                    rejected += 1
                    continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                rejected += 1
                continue
            seen.add(digest)
            records.append({
                "id": f"{book.name}/{path}",
                "text": text,
                "metadata": {"book": book.name, "file": path,
                             "type": "python_source" if path.endswith(".py") else "prose",
                             "license": book.license},
            })

        with (output_dir / f"{book.name}.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        chars = sum(len(r["text"]) for r in records)
        total_docs += len(records)
        total_chars += chars
        provenance.append({"name": book.name, "repository": f"https://github.com/{book.repository}",
                           "revision": book.revision, "archive_sha256": hashlib.sha256(archive).hexdigest(),
                           "license": book.license, "license_file": f"licenses/{book.name}_{PurePosixPath(book.license_file).name}",
                           "documents": len(records), "characters": chars, "rejected": rejected})
        print(f"{book.name:<22} {len(records):>5} documents  {chars:>10,} characters  ({rejected} rejected)")

    (output_dir / "provenance.json").write_text(json.dumps(
        {"description": "Openly licensed (shareable) Python books and courses.",
         "document_count": total_docs, "corpus_characters": total_chars, "sources": provenance},
        indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {total_docs} documents and {total_chars:,} characters to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "sources" / "python_books")
    build(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
