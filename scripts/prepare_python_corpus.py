#!/usr/bin/env python3
"""Build a revision-pinned Python documentation and source-code corpus.

The allowlist deliberately uses projects with permissive licenses. Archives
are downloaded at exact Git commit IDs, relevant English documentation and
Python files are selected, Python syntax is checked, exact duplicates are
removed, and every upstream license is retained beside the corpus.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import re
import tarfile
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


USER_AGENT = "EducationalPythonCorpusBuilder/1.0"
TEXT_EXTENSIONS = {".py", ".md", ".rst"}
SKIP_PARTS = {
    ".github",
    ".gitlab",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "site-packages",
    "test",
    "tests",
    "testing",
    "vendor",
    "vendored",
    "_vendor",
}


@dataclass(frozen=True)
class Source:
    name: str
    repository: str
    revision: str
    spdx_license: str
    license_fragment: str
    patterns: tuple[str, ...]
    kind: str

    @property
    def archive_url(self) -> str:
        return f"https://codeload.github.com/{self.repository}/tar.gz/{self.revision}"

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"


SOURCES = (
    Source(
        "cpython",
        "python/cpython",
        "02fae7a9594953c1d8b298b08d6faa99a5c18975",
        "PSF-2.0",
        "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2",
        (
            "Doc/tutorial/*.rst",
            "Doc/howto/*.rst",
            "Doc/library/*.rst",
            "Doc/reference/*.rst",
            "Doc/extending/*.rst",
            "Doc/c-api/*.rst",
            "Lib/*.py",
            "Lib/asyncio/*.py",
            "Lib/collections/*.py",
            "Lib/concurrent/*.py",
            "Lib/concurrent/futures/*.py",
            "Lib/email/*.py",
            "Lib/email/*/*.py",
            "Lib/http/*.py",
            "Lib/importlib/*.py",
            "Lib/importlib/*/*.py",
            "Lib/json/*.py",
            "Lib/logging/*.py",
            "Lib/sqlite3/*.py",
            "Lib/unittest/*.py",
            "Lib/urllib/*.py",
            "Lib/xml/*.py",
            "Lib/xml/*/*.py",
        ),
        "official_documentation_and_standard_library",
    ),
    Source(
        "python_peps",
        "python/peps",
        "96fff83778298b0a015cbefbec12781b8f0bf6d5",
        "CC0-1.0 OR Public-Domain",
        "CC0 1.0 Universal",
        ("peps/pep-*.rst",),
        "language_design_documents",
    ),
    Source(
        "flask",
        "pallets/flask",
        "d73fa1cdcbd8b1465c151db8924ba58b1dd14e35",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ("src/flask/*.py", "docs/*.rst", "docs/*/*.rst", "examples/*.py", "examples/*/*.py"),
        "library_source_and_documentation",
    ),
    Source(
        "click",
        "pallets/click",
        "3cbaa76b6014be7427d1ab06b4af40f01e7c278e",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ("src/click/*.py", "docs/*.rst", "docs/*/*.rst", "examples/*.py", "examples/*/*.py"),
        "library_source_and_documentation",
    ),
    Source(
        "requests",
        "psf/requests",
        "611c6162cbc4ac2020a2f91c7cfa4f3abf9bbb60",
        "Apache-2.0",
        "Apache License",
        ("src/requests/*.py", "docs/*.rst", "docs/*/*.rst"),
        "library_source_and_documentation",
    ),
    Source(
        "rich",
        "Textualize/rich",
        "9d8f9a372cc5916fd4781fec207ced7ddac2f08f",
        "MIT",
        "Permission is hereby granted, free of charge",
        ("rich/*.py", "docs/*.md", "docs/*/*.md", "examples/*.py", "README.md", "FAQ.md"),
        "library_source_and_documentation",
    ),
    Source(
        "attrs",
        "python-attrs/attrs",
        "8f767776326faaed11e6c2974798787f6e19b343",
        "MIT",
        "Permission is hereby granted, free of charge",
        ("src/attr/*.py", "src/attrs/*.py", "docs/*.md", "docs/*.rst", "docs/*/*.md", "docs/*/*.rst"),
        "library_source_and_documentation",
    ),
    Source(
        "fastapi",
        "fastapi/fastapi",
        "50113da16fec53b66b80d75e80a89296de4fa5a5",
        "MIT",
        "Permission is hereby granted, free of charge",
        ("fastapi/*.py", "fastapi/*/*.py", "docs/en/*.md", "docs/en/*/*.md", "docs_src/*.py", "docs_src/*/*.py", "docs_src/*/*/*.py"),
        "library_source_and_documentation",
    ),
    Source(
        "black",
        "psf/black",
        "debb3169c698144c6b879da12b340d393c4fd0b5",
        "MIT",
        "Permission is hereby granted, free of charge",
        ("src/black/*.py", "src/black/*/*.py", "docs/*.md", "docs/*/*.md"),
        "tool_source_and_documentation",
    ),
)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def relative_archive_path(member_name: str) -> str | None:
    path = PurePosixPath(member_name)
    if len(path.parts) < 2 or path.is_absolute() or ".." in path.parts:
        return None
    return PurePosixPath(*path.parts[1:]).as_posix()


def is_selected(path: str, patterns: tuple[str, ...]) -> bool:
    pure_path = PurePosixPath(path)
    if pure_path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    if any(part.lower() in SKIP_PARTS for part in pure_path.parts):
        return False
    if any(part.startswith(".") for part in pure_path.parts):
        return False
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def normalize_text(raw: bytes) -> str | None:
    if b"\x00" in raw:
        return None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE).strip()
    if len(text) < 200 or len(text) > 250_000:
        return None
    return text + "\n"


def find_license(files: dict[str, bytes], source: Source) -> tuple[str, bytes]:
    if source.name == "python_peps":
        policy_path = "peps/pep-0001.rst"
        policy = files.get(policy_path)
        if policy is None or b"CC0-1.0-Universal" not in policy:
            raise ValueError("Could not verify the PEP 1 public-domain/CC0 policy")
        return policy_path, policy
    candidates = []
    for path, content in files.items():
        pure_path = PurePosixPath(path)
        if len(pure_path.parts) == 1 and pure_path.name.lower().startswith(("license", "copying")):
            candidates.append((path, content))
    for path, content in sorted(candidates):
        decoded = content.decode("utf-8", errors="replace")
        if source.license_fragment.lower() in decoded.lower():
            return path, content
    raise ValueError(f"Could not verify the declared license for {source.name}")


def pep_is_open(text: str) -> bool:
    # PEPs put their license declaration at the end. Restricting the check to
    # the tail avoids accepting a document that merely discusses CC0 or the
    # public domain in its main text.
    lowered = text[-3_000:].lower()
    return "public domain" in lowered or "cc0-1.0" in lowered or "cc0 1.0" in lowered


def document(source: Source, path: str, text: str) -> str:
    language = "Python source" if path.endswith(".py") else "Technical documentation"
    return (
        "===== PYTHON CORPUS DOCUMENT =====\n"
        f"Project: {source.name}\n"
        f"File: {path}\n"
        f"Type: {language}\n\n"
        f"{text.rstrip()}"
    )


def rebuild_combined(output_dir: Path) -> None:
    literature_path = output_dir / "corpus.txt"
    python_path = output_dir / "python_corpus.txt"
    if not literature_path.exists() or not python_path.exists():
        return
    literature = literature_path.read_text(encoding="utf-8").rstrip()
    python = python_path.read_text(encoding="utf-8").rstrip()
    combined = literature + "\n\n\n===== PYTHON TRAINING MATERIAL =====\n\n\n" + python + "\n"
    (output_dir / "combined_corpus.txt").write_text(combined, encoding="utf-8")


def build(output_dir: Path) -> None:
    licenses_dir = output_dir / "python_licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    documents: list[str] = []
    provenance_sources = []
    total_rejections = {"syntax": 0, "duplicate": 0, "format_or_size": 0}

    for source in SOURCES:
        archive = fetch(source.archive_url)
        files: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member in tar:
                if not member.isfile() or member.issym() or member.islnk():
                    continue
                relative = relative_archive_path(member.name)
                if relative is None or member.size > 1_000_000:
                    continue
                extracted = tar.extractfile(member)
                if extracted is not None:
                    files[relative] = extracted.read()

        license_path, license_bytes = find_license(files, source)
        license_destination = licenses_dir / f"{source.name}_{PurePosixPath(license_path).name}"
        license_destination.write_bytes(license_bytes)

        source_documents = []
        source_characters = 0
        source_python_files = 0
        source_documentation_files = 0
        for path in sorted(files):
            if not is_selected(path, source.patterns):
                continue
            text = normalize_text(files[path])
            if text is None:
                total_rejections["format_or_size"] += 1
                continue
            if source.name == "python_peps" and not pep_is_open(text):
                total_rejections["format_or_size"] += 1
                continue
            if path.endswith(".py"):
                try:
                    ast.parse(text, filename=path)
                except SyntaxError:
                    total_rejections["syntax"] += 1
                    continue
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                total_rejections["duplicate"] += 1
                continue
            seen_hashes.add(content_hash)
            rendered = document(source, path, text)
            source_documents.append(rendered)
            source_characters += len(rendered)
            if path.endswith(".py"):
                source_python_files += 1
            else:
                source_documentation_files += 1

        documents.extend(source_documents)
        provenance_sources.append(
            {
                "name": source.name,
                "repository": source.repository_url,
                "revision": source.revision,
                "archive_url": source.archive_url,
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
                "license": source.spdx_license,
                "license_file": f"python_licenses/{license_destination.name}",
                "kind": source.kind,
                "documents": len(source_documents),
                "python_files": source_python_files,
                "documentation_files": source_documentation_files,
                "characters": source_characters,
            }
        )
        print(
            f"{source.name:<12} {len(source_documents):>5} documents  "
            f"{source_characters:>10,} characters"
        )

    corpus = "\n\n\n".join(documents) + "\n"
    (output_dir / "python_corpus.txt").write_text(corpus, encoding="utf-8")
    provenance = {
        "description": "Revision-pinned permissively licensed Python documentation and source code.",
        "selection_policy": "English documentation and parseable Python from an explicit source/path allowlist; tests, vendored code, generated assets, and exact duplicates are excluded.",
        "source_count": len(SOURCES),
        "document_count": len(documents),
        "corpus_characters": len(corpus),
        "rejections": total_rejections,
        "sources": provenance_sources,
    }
    (output_dir / "python_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rebuild_combined(output_dir)
    print(f"\nWrote {len(documents)} documents and {len(corpus):,} characters to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "pretrain",
    )
    args = parser.parse_args()
    build(args.output_dir)


if __name__ == "__main__":
    main()
