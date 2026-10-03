#!/usr/bin/env python3
"""Build a revision-pinned Rust and C++ educational code corpus.

The corpus combines explanatory Rust material with real Rust listings and two
independent kinds of modern C++: a production formatting library and a broad
collection of educational algorithms.  Every upstream revision is pinned,
license text is retained, and exact duplicate files are removed.

Output is ``data/sources/code_docs/*.jsonl``, ready for build_mix.py.
"""

from __future__ import annotations

import argparse
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


USER_AGENT = "BobGPTEducationalCorpusBuilder/1.0"
SKIP_PARTS = {
    ".git",
    ".github",
    "build",
    "target",
    "node_modules",
    "test",
    "tests",
    "testing",
    "vendor",
    "vendored",
}


@dataclass(frozen=True)
class Source:
    name: str
    repository: str
    revision: str
    language: str
    spdx_license: str
    license_names: tuple[str, ...]
    patterns: tuple[str, ...]
    kind: str

    @property
    def archive_url(self) -> str:
        return f"https://codeload.github.com/{self.repository}/tar.gz/{self.revision}"


SOURCES = (
    Source(
        name="rust_book",
        repository="rust-lang/book",
        revision="1500248d8f230566e4ec9f27fcbb8fe9e2898ab1",
        language="rust",
        spdx_license="MIT OR Apache-2.0",
        license_names=("LICENSE-MIT", "LICENSE-APACHE"),
        patterns=("src/*.md", "listings/*.rs", "listings/*.toml"),
        kind="official_language_book_and_runnable_listings",
    ),
    Source(
        name="fmt",
        repository="fmtlib/fmt",
        revision="5da4e9a3fb15626ba92969199061e0169ebc8a00",
        language="cpp",
        spdx_license="MIT",
        license_names=("LICENSE",),
        patterns=("include/fmt/*.h", "src/*.cc", "doc/*.md", "doc/*.rst", "README.md"),
        kind="production_library_source_and_documentation",
    ),
    Source(
        name="cpp_algorithms",
        repository="TheAlgorithms/C-Plus-Plus",
        revision="6d81fa37622b89811d51319ce3d39abf5c217baf",
        language="cpp",
        spdx_license="MIT",
        license_names=("LICENSE",),
        patterns=("*.cpp", "*.cc", "*.h", "*.hpp", "README.md", "CodingGuidelines.md"),
        kind="educational_algorithms_and_data_structures",
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
    pure = PurePosixPath(path)
    if any(part.lower() in SKIP_PARTS for part in pure.parts):
        return False
    if any(part.startswith(".") for part in pure.parts):
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
    if len(text) < 120 or len(text) > 500_000:
        return None
    return text + "\n"


def load_archive(source: Source) -> tuple[bytes, dict[str, bytes]]:
    archive = fetch(source.archive_url)
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.issym() or member.islnk() or member.size > 2_000_000:
                continue
            relative = relative_archive_path(member.name)
            if relative is None:
                continue
            extracted = tar.extractfile(member)
            if extracted is not None:
                files[relative] = extracted.read()
    return archive, files


def document(source: Source, path: str, text: str) -> dict:
    extension = PurePosixPath(path).suffix.lower()
    content_type = "documentation" if extension in {".md", ".rst"} else "source_code"
    return {
        "id": f"{source.name}/{path}",
        "text": text,
        "metadata": {
            "project": source.name,
            "repository": f"https://github.com/{source.repository}",
            "revision": source.revision,
            "file": path,
            "language": source.language,
            "type": content_type,
            "license": source.spdx_license,
        },
    }


def build(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    licenses_dir = output_dir / "licenses"
    licenses_dir.mkdir(exist_ok=True)
    seen_hashes: set[str] = set()
    provenance_sources = []
    total_documents = total_characters = total_duplicates = 0

    for source in SOURCES:
        archive, files = load_archive(source)
        license_files = []
        for license_name in source.license_names:
            license_bytes = files.get(license_name)
            if license_bytes is None:
                raise ValueError(f"{source.name}: missing required {license_name}")
            destination = licenses_dir / f"{source.name}_{license_name}"
            destination.write_bytes(license_bytes)
            license_files.append(f"licenses/{destination.name}")

        records = []
        characters = duplicates = rejected = 0
        for path in sorted(files):
            if not is_selected(path, source.patterns):
                continue
            text = normalize_text(files[path])
            if text is None:
                rejected += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(digest)
            records.append(document(source, path, text))
            characters += len(text)

        with (output_dir / f"{source.name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        provenance_sources.append(
            {
                "name": source.name,
                "repository": f"https://github.com/{source.repository}",
                "revision": source.revision,
                "archive_url": source.archive_url,
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
                "language": source.language,
                "license": source.spdx_license,
                "license_files": license_files,
                "kind": source.kind,
                "documents": len(records),
                "characters": characters,
                "exact_duplicates_removed": duplicates,
                "format_or_size_rejections": rejected,
            }
        )
        total_documents += len(records)
        total_characters += characters
        total_duplicates += duplicates
        print(f"{source.name:<16} {len(records):>5,} documents  {characters:>12,} characters")

    provenance = {
        "description": "Revision-pinned educational Rust and C++ documentation and source code.",
        "selection_policy": "Explicit project/path allowlist; tests, build output, vendored material, oversized files, and exact duplicates are excluded.",
        "source_count": len(SOURCES),
        "document_count": total_documents,
        "corpus_characters": total_characters,
        "exact_duplicates_removed": total_duplicates,
        "sources": provenance_sources,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {total_documents:,} documents and {total_characters:,} characters to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sources" / "code_docs",
    )
    build(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
