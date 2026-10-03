#!/usr/bin/env python3
"""Build a revision-pinned Python documentation and source-code corpus.

The allowlist deliberately uses projects with permissive licenses. Archives
are downloaded at exact Git commit IDs, relevant English documentation and
Python files are selected, Python syntax is checked, exact duplicates are
removed, and every upstream license is retained beside the corpus.

Output is data/sources/python_docs/<project>.jsonl, one record per file, ready
for scripts/build_mix.py.
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
TEXT_EXTENSIONS = {".py", ".md", ".rst", ".txt"}  # .txt only where a pattern asks for it (Django docs)
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
    # Added 2026-09-29: more permissive Python projects instead of repeating python_docs.
    Source(
        "numpy",
        "numpy/numpy",
        "5176335c709e27ad07589aecfaef9491f5097b83",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('numpy/*.py', 'doc/source/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "pandas",
        "pandas-dev/pandas",
        "1fb9ce7653775227975f29af55387170a59ce2db",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('pandas/*.py', 'doc/source/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "django",
        "django/django",
        "5a4511adb247a44a1cada11fe4763abba0a42663",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('django/*.py', 'docs/*.txt'),
        "library_source_and_documentation",
    ),
    Source(
        "sqlalchemy",
        "sqlalchemy/sqlalchemy",
        "bc8982869d887a47a535ba45d80eb9a6f4595530",
        "MIT",
        "Permission is hereby granted, free of charge",
        ('lib/sqlalchemy/*.py', 'doc/build/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "pytest",
        "pytest-dev/pytest",
        "2887015cade4757385308e7a7d8083557fc637e2",
        "MIT",
        "Permission is hereby granted, free of charge",
        ('src/*.py', 'doc/en/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "httpx",
        "encode/httpx",
        "b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('httpx/*.py', 'docs/*.md'),
        "library_source_and_documentation",
    ),
    Source(
        "pydantic",
        "pydantic/pydantic",
        "26f7b8a1ad02951962c4073297fa9a2b1ce8d30c",
        "MIT",
        "Permission is hereby granted, free of charge",
        ('pydantic/*.py', 'docs/*.md'),
        "library_source_and_documentation",
    ),
    Source(
        "scikit_learn",
        "scikit-learn/scikit-learn",
        "c9021ac077341eef99fe4c1756597f52d3b2a870",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('sklearn/*.py', 'doc/*.rst', 'examples/*.py'),
        "library_source_and_documentation",
    ),
    Source(
        "jinja",
        "pallets/jinja",
        "5ef70112a1ff19c05324ff889dd30405b1002044",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('src/jinja2/*.py', 'docs/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "werkzeug",
        "pallets/werkzeug",
        "f7e37f0bf510fa355fdac3e922cc2078916b021f",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('src/werkzeug/*.py', 'docs/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "markupsafe",
        "pallets/markupsafe",
        "b2e4d9c7687be25695fffbe93a37622302b24fb1",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('src/markupsafe/*.py', 'docs/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "typer",
        "fastapi/typer",
        "a80f6e5ecd74f32b983cca336a2f3cba98d9853a",
        "MIT",
        "Permission is hereby granted, free of charge",
        ('typer/*.py', 'docs/*.md', 'docs_src/*.py'),
        "library_source_and_documentation",
    ),
    Source(
        "aiohttp",
        "aio-libs/aiohttp",
        "3489636c044df3176665a2173d91ddef6d8161ec",
        "Apache-2.0",
        "Apache License",
        ('aiohttp/*.py', 'docs/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "mypy",
        "python/mypy",
        "92d831537bb7d0df0f98907f154eaf736e7b60ae",
        "MIT",
        "Permission is hereby granted, free of charge",
        ('mypy/*.py', 'docs/source/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "pip",
        "pypa/pip",
        "a7002c9771a6c3f0317a4e6b9fbdcd22e643f7b6",
        "MIT",
        "Permission is hereby granted, free of charge",
        ('src/pip/*.py', 'docs/html/*.md', 'docs/html/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "sphinx",
        "sphinx-doc/sphinx",
        "b04a2101295ac3fb725b16111eda0284b6da4cca",
        "BSD-2-Clause",
        "Redistribution and use in source and binary forms",
        ('sphinx/*.py', 'doc/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "celery",
        "celery/celery",
        "440e4e4ba144e4f10efc039a46cc2321a9a61d0b",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('celery/*.py', 'docs/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "networkx",
        "networkx/networkx",
        "92f497e2eb8192d1ce9205595f512294e4a9b696",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('networkx/*.py', 'doc/*.rst', 'examples/*.py'),
        "library_source_and_documentation",
    ),
    Source(
        "more_itertools",
        "more-itertools/more-itertools",
        "1ea82a711c69f590054987b5cb194157f8ce8ac4",
        "MIT",
        "Permission is hereby granted, free of charge",
        ('more_itertools/*.py', 'docs/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "httpie",
        "httpie/cli",
        "5b604c37c6c67e18e7c3e9aee6c88a8c22b98345",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('httpie/*.py', 'docs/*.md'),
        "library_source_and_documentation",
    ),
    Source(
        "scrapy",
        "scrapy/scrapy",
        "5d789b24f994bcdfca97c34d976c73739e064849",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('scrapy/*.py', 'docs/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "sympy",
        "sympy/sympy",
        "ce96d33116cc816c02fe759200b3424ffcaa98e1",
        "BSD-3-Clause",
        "Redistribution and use in source and binary forms",
        ('sympy/*.py', 'doc/src/*.md', 'doc/src/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "cattrs",
        "python-attrs/cattrs",
        "0a1d658438f1cd0d2a409edf867c0f1d3ef40398",
        "MIT",
        "Permission is hereby granted, free of charge",
        ('src/cattrs/*.py', 'docs/*.md', 'docs/*.rst'),
        "library_source_and_documentation",
    ),
    Source(
        "textual",
        "Textualize/textual",
        "06dbeef4bb70fb718236aa418ed658ef4667a126",
        "MIT",
        "Permission is hereby granted, free of charge",
        ('src/textual/*.py', 'docs/*.md', 'examples/*.py'),
        "library_source_and_documentation",
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


def document(source: Source, path: str, text: str) -> dict:
    return {
        "id": f"{source.name}/{path}",
        "text": text,
        "metadata": {
            "project": source.name,
            "file": path,
            "type": "python_source" if path.endswith(".py") else "documentation",
            "license": source.spdx_license,
        },
    }


def build(output_dir: Path) -> None:
    licenses_dir = output_dir / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    document_count = 0
    corpus_characters = 0
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
            source_documents.append(document(source, path, text))
            source_characters += len(text)
            if path.endswith(".py"):
                source_python_files += 1
            else:
                source_documentation_files += 1

        with (output_dir / f"{source.name}.jsonl").open("w", encoding="utf-8") as f:
            for record in source_documents:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        document_count += len(source_documents)
        corpus_characters += source_characters
        provenance_sources.append(
            {
                "name": source.name,
                "repository": source.repository_url,
                "revision": source.revision,
                "archive_url": source.archive_url,
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
                "license": source.spdx_license,
                "license_file": f"licenses/{license_destination.name}",
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

    provenance = {
        "description": "Revision-pinned permissively licensed Python documentation and source code.",
        "selection_policy": "English documentation and parseable Python from an explicit source/path allowlist; tests, vendored code, generated assets, and exact duplicates are excluded.",
        "source_count": len(SOURCES),
        "document_count": document_count,
        "corpus_characters": corpus_characters,
        "rejections": total_rejections,
        "sources": provenance_sources,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {document_count} documents and {corpus_characters:,} characters to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sources" / "python_docs",
    )
    args = parser.parse_args()
    build(args.output_dir)


if __name__ == "__main__":
    main()
