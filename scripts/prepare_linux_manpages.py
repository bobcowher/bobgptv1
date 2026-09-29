#!/usr/bin/env python3
"""Build a plain-text Linux interfaces corpus from a verified man-pages release.

This is the kernel-userspace API reference maintained by the Linux man-pages
project.  It includes system calls, C library interfaces, file formats,
protocols, administration interfaces, and capabilities(7).  Alias pages that
contain only a roff ``.so`` redirect are omitted to avoid duplicates.

The script requires ``groff`` and ``col`` to render roff into readable UTF-8.
Output is ``data/sources/linux_manpages/manpages.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import unicodedata
import urllib.request
from pathlib import Path, PurePosixPath


VERSION = "6.19"
ARCHIVE_NAME = f"man-pages-{VERSION}.tar.xz"
URL = f"https://cdn.kernel.org/pub/linux/docs/man-pages/{ARCHIVE_NAME}"
ARCHIVE_SHA256 = "88a7c42ad2e03d8b96dc72d95e451f2d875ff0f43103a8eb8ac8242133bdcb05"
USER_AGENT = "BobGPTEducationalCorpusBuilder/1.0"


def fetch() -> bytes:
    request = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive = response.read()
    actual = hashlib.sha256(archive).hexdigest()
    if actual != ARCHIVE_SHA256:
        raise ValueError(f"archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual}")
    return archive


def relative_path(member_name: str) -> str | None:
    path = PurePosixPath(member_name)
    if len(path.parts) < 3 or path.is_absolute() or ".." in path.parts:
        return None
    relative = PurePosixPath(*path.parts[1:])
    # Since 6.19 the release stores pages under man/manN rather than manN.
    if relative.parts and relative.parts[0] == "man":
        relative = PurePosixPath(*relative.parts[1:])
    return relative.as_posix()


def render(raw: bytes, path: str) -> str | None:
    # Alias pages may contain copyright comments before their sole .so redirect.
    meaningful = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith(b'.\\"')
    ]
    if len(meaningful) == 1 and meaningful[0].startswith(b".so "):
        return None
    groff = subprocess.run(
        ["groff", "-Kutf8", "-mandoc", "-Tutf8"],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GROFF_NO_SGR": "1"},
        check=False,
    )
    if groff.returncode:
        error = groff.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"groff failed for {path}: {error}")
    col = subprocess.run(
        ["col", "-b"], input=groff.stdout, stdout=subprocess.PIPE, check=True
    )
    text = col.stdout.decode("utf-8", errors="strict")
    # grotty emits SGR underline/bold sequences even after overstrikes are
    # removed by col. They are terminal presentation, not document content.
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = unicodedata.normalize("NFC", text).replace("\f", "\n")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    if len(text) < 120:
        return None
    return text + "\n"


def build(output_dir: Path) -> None:
    for executable in ("groff", "col"):
        if shutil.which(executable) is None:
            raise SystemExit(f"{executable!r} is required to render Linux man pages")

    archive = fetch()
    records = []
    aliases = 0
    license_files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as tar:
        for member in tar:
            if not member.isfile() or member.issym() or member.islnk():
                continue
            path = relative_path(member.name)
            if path is None:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read()
            if path.startswith("LICENSES/"):
                license_files[path] = raw
                continue
            if not re.fullmatch(r"man[1-8]/[^/]+\.[1-8][a-z]*", path):
                continue
            text = render(raw, path)
            if text is None:
                aliases += 1
                continue
            filename = PurePosixPath(path).name
            name, section = filename.rsplit(".", 1)
            records.append(
                {
                    "id": f"linux_manpages/{path}",
                    "text": text,
                    "metadata": {
                        "name": name,
                        "section": section,
                        "source_file": path,
                        "project": "Linux man-pages",
                        "version": VERSION,
                        "source_url": URL,
                        "license": "per-page; see retained LICENSES directory and source header",
                    },
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    licenses_dir = output_dir / "licenses"
    licenses_dir.mkdir(exist_ok=True)
    for path, contents in license_files.items():
        destination = licenses_dir / PurePosixPath(path).name
        destination.write_bytes(contents)
    with (output_dir / "manpages.jsonl").open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item["id"]):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    provenance = {
        "description": "Rendered Linux kernel/userspace interface documentation from the Linux man-pages project.",
        "project_url": "https://www.kernel.org/doc/man-pages/",
        "version": VERSION,
        "archive_url": URL,
        "archive_sha256": ARCHIVE_SHA256,
        "document_count": len(records),
        "alias_pages_omitted": aliases,
        "corpus_characters": sum(len(record["text"]) for record in records),
        "rendering": "groff -Kutf8 -mandoc -Tutf8, then col -b",
        "license_policy": "Linux man-pages files carry per-page licenses; all release LICENSES files are retained.",
        "license_files": sorted(f"licenses/{path.name}" for path in licenses_dir.iterdir()),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(records):,} rendered pages ({provenance['corpus_characters']:,} characters); "
        f"omitted {aliases:,} aliases"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sources" / "linux_manpages",
    )
    build(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
