# Training data

This repository has two reproducible educational datasets under `data/`.
The generated data itself remains ignored by Git (as established by the
project's existing `.gitignore`); the generator scripts are tracked.

## Pretraining corpus

Run:

```bash
python scripts/prepare_public_domain_corpus.py
```

This creates:

- `data/pretrain/books/*.txt`: 25 cleaned books, one file per work.
- `data/pretrain/corpus.txt`: all books joined into one next-token-training
  corpus.
- `data/pretrain/provenance.json`: title, author, genre, publication year,
  source links, rights status, character counts, and SHA-256 hashes.

The selection emphasizes foundational science fiction and speculative fiction
while including several works with early modernist or Jazz Age prose. Before a
download is accepted, the script checks the book's current Project Gutenberg
RDF record for the exact statement `Public domain in the USA.`. It then removes
the Project Gutenberg header and license footer from the training copy.

That status is specific to the United States. Check copyright law in the place
where the data will be downloaded, distributed, or used. Project Gutenberg's
policy is at <https://www.gutenberg.org/policy/license>.

### Python pretraining material

After preparing the literature corpus, run:

```bash
python scripts/prepare_python_corpus.py
```

This creates:

- `data/pretrain/python_corpus.txt`: filtered Python source and technical
  documentation.
- `data/pretrain/python_provenance.json`: repositories, exact Git revisions,
  archive hashes, licenses, selection counts, and rejection counts.
- `data/pretrain/python_licenses/`: a copy of every upstream license or license
  policy used by the corpus.
- `data/pretrain/combined_corpus.txt`: literature followed by the Python
  corpus, ready for the current next-token training pipeline.

The Python sources are deliberately limited to an explicit permissive-license
allowlist: CPython documentation and selected standard-library modules, PEPs
whose final license section declares public-domain or CC0 terms, and selected
MIT/BSD/Apache projects. The builder pins exact commits, excludes tests and
vendored/generated material, rejects invalid Python, and removes exact
duplicates. Preserve `python_provenance.json` and `python_licenses/` when
redistributing the corpus or a derived dataset.

## Python Q&A corpus

Run:

```bash
python scripts/generate_python_qa.py
```

This creates deterministic training and validation splits in two forms:

- `data/finetune/{train,validation}.jsonl`: chat records with `system`, `user`,
  and `assistant` messages.
- `data/finetune/{train,validation}.txt`: the same records serialized with
  `### System`, `### Question`, `### Answer`, and `### End` markers for a plain
  causal-language-model loader.
- `data/finetune/metadata.json`: counts and format information.

The examples cover Python concepts, output prediction, function writing, and
debugging. The generator checks IDs and pairs for duplicates and parses every
Python fenced block with `ast.parse`. The synthetic examples are released under
CC0-1.0.

`scripts/train.py` loads `data/pretrain/combined_corpus.txt`. The loader
tokenizes it once into one contiguous tensor, splits that tensor into training
and validation views, and creates shifted inputs and targets only as batches
are requested. The plain-text Q&A files can be consumed by the same next-token
objective, while the JSONL form is intended for a future instruction-aware
data loader.
