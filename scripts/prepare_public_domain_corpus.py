#!/usr/bin/env python3
"""Download, verify, and clean a small public-domain literature corpus.

Every source is a Project Gutenberg plain-text edition.  Before accepting a
book, the script reads its RDF catalog record and requires the exact rights
statement "Public domain in the USA.".  Gutenberg's header and license footer
are removed from the training copy; source URLs and content hashes are kept in
provenance.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


SOURCES = [
    # Foundational and early-20th-century science/speculative fiction.
    (35, "the_time_machine", "The Time Machine", "H. G. Wells", 1895, "science_fiction"),
    (36, "the_war_of_the_worlds", "The War of the Worlds", "H. G. Wells", 1898, "science_fiction"),
    (5230, "the_invisible_man", "The Invisible Man", "H. G. Wells", 1897, "science_fiction"),
    (159, "the_island_of_doctor_moreau", "The Island of Doctor Moreau", "H. G. Wells", 1896, "science_fiction"),
    (775, "when_the_sleeper_wakes", "When the Sleeper Wakes", "H. G. Wells", 1899, "science_fiction"),
    (52501, "the_first_men_in_the_moon", "The First Men in the Moon", "H. G. Wells", 1901, "science_fiction"),
    (62, "a_princess_of_mars", "A Princess of Mars", "Edgar Rice Burroughs", 1912, "science_fiction"),
    (64, "the_gods_of_mars", "The Gods of Mars", "Edgar Rice Burroughs", 1913, "science_fiction"),
    (68, "the_warlord_of_mars", "The Warlord of Mars", "Edgar Rice Burroughs", 1914, "science_fiction"),
    (72, "thuvia_maid_of_mars", "Thuvia, Maid of Mars", "Edgar Rice Burroughs", 1916, "science_fiction"),
    (96, "the_monster_men", "The Monster Men", "Edgar Rice Burroughs", 1913, "science_fiction"),
    (139, "the_lost_world", "The Lost World", "Arthur Conan Doyle", 1912, "science_fiction"),
    (84, "frankenstein", "Frankenstein", "Mary Wollstonecraft Shelley", 1818, "science_fiction"),
    (18247, "the_last_man", "The Last Man", "Mary Wollstonecraft Shelley", 1826, "science_fiction"),
    (164, "twenty_thousand_leagues", "Twenty Thousand Leagues under the Sea", "Jules Verne", 1870, "science_fiction"),
    (83, "from_the_earth_to_the_moon", "From the Earth to the Moon", "Jules Verne", 1865, "science_fiction"),
    (18857, "a_journey_to_the_centre_of_the_earth", "A Journey to the Centre of the Earth", "Jules Verne", 1864, "science_fiction"),
    (201, "flatland", "Flatland", "Edwin A. Abbott", 1884, "speculative_fiction"),
    (61963, "we", "We", "Yevgeny Zamyatin; translated by Gregory Zilboorg", 1924, "science_fiction"),
    (68283, "the_call_of_cthulhu", "The Call of Cthulhu", "H. P. Lovecraft", 1928, "speculative_fiction"),
    # More modern prose styles that are now public domain in the United States.
    (64317, "the_great_gatsby", "The Great Gatsby", "F. Scott Fitzgerald", 1925, "modern_fiction"),
    (71865, "mrs_dalloway", "Mrs. Dalloway", "Virginia Woolf", 1925, "modern_fiction"),
    (61085, "in_our_time", "In Our Time", "Ernest Hemingway", 1924, "modern_fiction"),
    (805, "this_side_of_paradise", "This Side of Paradise", "F. Scott Fitzgerald", 1920, "modern_fiction"),
    (9830, "the_beautiful_and_damned", "The Beautiful and Damned", "F. Scott Fitzgerald", 1922, "modern_fiction"),
]

RIGHTS_REQUIRED = "Public domain in the USA."
USER_AGENT = "EducationalCorpusBuilder/1.0 (Project Gutenberg downloader)"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def rdf_values(raw_rdf: bytes, suffix: str) -> list[str]:
    root = ET.fromstring(raw_rdf)
    return [
        " ".join("".join(element.itertext()).split())
        for element in root.iter()
        if element.tag.endswith(suffix)
    ]


def clean_gutenberg_text(raw: bytes, ebook_id: int) -> str:
    text = raw.decode("utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    start = re.search(
        r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    end = re.search(
        r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not start or not end or end.start() <= start.end():
        raise ValueError(f"Could not locate Gutenberg wrapper markers for ebook {ebook_id}")

    text = text[start.end() : end.start()]
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = text.strip() + "\n"
    if len(text) < 5_000:
        raise ValueError(f"Suspiciously short cleaned text for ebook {ebook_id}")
    return text


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rebuild_combined(output_dir: Path) -> None:
    python_path = output_dir / "python_corpus.txt"
    literature_path = output_dir / "corpus.txt"
    if not python_path.exists() or not literature_path.exists():
        return
    literature = literature_path.read_text(encoding="utf-8").rstrip()
    python = python_path.read_text(encoding="utf-8").rstrip()
    combined = literature + "\n\n\n===== PYTHON TRAINING MATERIAL =====\n\n\n" + python + "\n"
    (output_dir / "combined_corpus.txt").write_text(combined, encoding="utf-8")


def build(output_dir: Path) -> None:
    books_dir = output_dir / "books"
    books_dir.mkdir(parents=True, exist_ok=True)
    provenance = []
    corpus_parts = []

    for ebook_id, slug, title, author, publication_year, genre in SOURCES:
        catalog_url = f"https://www.gutenberg.org/ebooks/{ebook_id}"
        rdf_url = f"{catalog_url}.rdf"
        text_url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"

        raw_rdf = fetch(rdf_url)
        rights = rdf_values(raw_rdf, "rights")
        languages = rdf_values(raw_rdf, "language")
        if RIGHTS_REQUIRED not in rights:
            raise ValueError(
                f"Refusing ebook {ebook_id}: expected {RIGHTS_REQUIRED!r}, found {rights!r}"
            )
        if "en" not in languages:
            raise ValueError(f"Refusing non-English ebook {ebook_id}: {languages!r}")

        raw_text = fetch(text_url)
        cleaned = clean_gutenberg_text(raw_text, ebook_id)
        filename = f"pg{ebook_id}_{slug}.txt"
        destination = books_dir / filename
        destination.write_text(cleaned, encoding="utf-8")

        corpus_parts.append(f"{title}\nby {author}\n\n{cleaned.rstrip()}")
        provenance.append(
            {
                "gutenberg_id": ebook_id,
                "title": title,
                "author": author,
                "publication_year": publication_year,
                "genre": genre,
                "rights": RIGHTS_REQUIRED,
                "jurisdiction": "United States",
                "catalog_url": catalog_url,
                "text_url": text_url,
                "file": f"books/{filename}",
                "raw_sha256": sha256(raw_text),
                "clean_sha256": sha256(cleaned.encode("utf-8")),
                "characters": len(cleaned),
            }
        )
        print(f"{ebook_id:>5}  {len(cleaned):>9,} chars  {title}")

    corpus = "\n\n\n===== NEXT BOOK =====\n\n\n".join(corpus_parts) + "\n"
    (output_dir / "corpus.txt").write_text(corpus, encoding="utf-8")
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "description": "Public-domain literature corpus for educational language-model pretraining.",
                "rights_scope": "Catalog records verified as public domain in the USA; check the law in your jurisdiction.",
                "source_policy": "https://www.gutenberg.org/policy/license",
                "book_count": len(provenance),
                "corpus_characters": len(corpus),
                "sources": provenance,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rebuild_combined(output_dir)
    print(f"\nWrote {len(provenance)} books and {len(corpus):,} characters to {output_dir}")


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
