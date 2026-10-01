# Training data

Data lives under `data/` (a symlink to `/data/datasets/bobgptv1`, ignored by
Git). The scripts that produce it and the mix configs are tracked.

```
data/
  sources/<source>/*.jsonl     one record per document; add data by adding a file
  build/<mix>/                 tokenized output of a mix; regenerate, never edit
    train.bin, val.bin         uint16 GPT-2 token ids, documents separated by <|endoftext|>
    manifest.json              the mix config it was built from + per-source counts
mixes/<mix>.json               (in Git) which sources go into a build, and how often
chat_template.py               (in Git) the only place chat messages become text
```

The pipeline is **sources → mix → build → train**:

```bash
python scripts/prepare_public_domain_corpus.py   # -> data/sources/books/
python scripts/prepare_python_corpus.py          # -> data/sources/python_docs/
python scripts/generate_python_qa.py             # -> data/sources/python_qa/
python scripts/prepare_code_corpus.py            # -> data/sources/code_docs/
python scripts/prepare_linux_manpages.py          # -> data/sources/linux_manpages/
python scripts/generate_code_linux_qa.py          # -> data/sources/{code,linux}_qa/
python scripts/generate_large_qa.py               # -> data/sources/{code,linux}_curriculum/
python scripts/prepare_python_books.py           # -> data/sources/python_books/
python scripts/prepare_fineweb_edu.py            # -> data/sources/fineweb_edu/
python scripts/build_mix.py pretrain_v5          # -> data/build/pretrain_v5/
scripts/sync_data.sh                             # push data/ to the lab box
```

`scripts/train.py` names a mix, and `make_loaders` in `dataset.py` memory-maps
its `.bin` files. It refuses to train if `mixes/<mix>.json` changed after the
last build.

## Source records

Every line is one JSON object with a unique `id` (unique within its source).
A record has either `text`:

```json
{"id": "cpython/Doc/library/json.rst", "text": "...", "metadata": {"project": "cpython", ...}}
```

or `messages`, for conversations:

```json
{"id": "python_tutor/python_concept_0001",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}],
 "metadata": {"category": "concept", ...}}
```

`messages` records are turned into text by `chat_template.render()` at build time.
The server needs to prompt with the same template.

Adding data means dropping a new `.jsonl` into a source directory, or adding a
new source directory, and listing it in a mix. Existing files are never edited
to add to them.

## Mixes

```json
{"val_fraction": 0.1, "chunk_chars": 20000,
 "sources": [{"name": "books", "repeat": 1}, {"name": "python_docs", "repeat": 1}]}
```

- `repeat` upsamples a source in train (integer copies). Val is never repeated.
- A source can set its own `val_fraction`. FineWeb-Edu uses 0.01 so its val
  stays ~2M tokens and the full-val pass each epoch stays short.
- **Train/val split:** a unit goes to val when `sha256(unit id)` lands in the
  bottom `val_fraction` of 1000 buckets. The split depends only on ids, so
  adding documents never moves existing ones between train and val.
- **Chunking:** documents longer than `chunk_chars` are cut at paragraph
  breaks into `<id>#partNNN` units before hashing. With only 25 books, hashing
  whole books would put 2-3 entire novels in val; hashing ~20K-character parts
  gives val prose from most books.
- Phases (e.g. pretraining vs. mixing in Q&A late) are just different mixes over
  the same sources. `pretrain_v1` is books + python_docs; `pretrain_v2` adds
  FineWeb-Edu and `python_qa` (repeat 5). Because the split depends only on
  ids, books/python_docs val units are identical in both, so a v2 model can be
  scored on `build/pretrain_v1/val.bin` for a like-for-like comparison.
- `pretrain_v5` (current) is v4 with python_docs grown from 9 to 33 projects
  (41M -> 139M characters; ~20% of train tokens) and 1,800 tutor_qa
  conversations. `scripts/eval_frozen.py` is pinned to the original 9 projects
  so its benchmark is unchanged.
- `pretrain_v4` drops the templated `*_curriculum` sources (14
  fill-in-the-blank templates; the model learns the template, not the
  subject) and adds `tutor_qa`. All hand-authored Q&A is repeated 3x.
- `pretrain_v3` removes the five identical copies of each Python Q&A record and
  adds each instructional record once. It adds real Rust/C++ educational code,
  Linux interface documentation, and a large distinct Python/Rust/C++/Linux
  worked-problem curriculum.

## Sources

### books: public-domain literature

25 Project Gutenberg works, science fiction and speculative fiction plus some
early modernist and Jazz Age prose. `gutenberg.jsonl` has one record per book.
`provenance.json` holds title, author, year, source links, rights status and
SHA-256 hashes.

Before a download is accepted, the script checks the book's Gutenberg RDF record
for the exact statement `Public domain in the USA.`, then strips the Gutenberg
header and license footer. That status is US-specific. Check the law where the
data will be used. See <https://www.gutenberg.org/policy/license>.

### python_docs: Python documentation and source

One `<project>.jsonl` per project. The projects are CPython docs and selected
stdlib modules, PEPs whose license section declares public domain or CC0, and
selected MIT/BSD/Apache projects. The builder pins exact commits, excludes
tests and vendored or generated material, rejects unparseable Python, and
removes exact duplicates. `provenance.json` records repositories, revisions,
archive hashes and rejection counts. `licenses/` keeps every upstream license.
Keep both when redistributing.

### fineweb_edu: educational web text

A ~200M-token slice of FineWeb-Edu `sample-10BT`
(<https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>, ODC-By 1.0):
Common Crawl pages filtered for educational value. The script downloads one
~750M-token parquet shard (cached by `huggingface_hub`) and keeps documents
whose id hashes into the requested fraction, so reruns are identical.
`--tokens` and `--shard` control size and which shard. Records keep url and
dump for attribution.

### python_qa: synthetic Python Q&A

`python_tutor.jsonl` has 208 deterministic chat examples: concepts, output
prediction, function writing and debugging. The generator checks for
duplicates and parses every fenced Python block with `ast.parse`. Released
under CC0-1.0.

### code_docs: Rust and C++ documentation/source

Revision-pinned snapshots of The Rust Programming Language (including runnable
listings), the MIT-licensed {fmt} library, and the MIT-licensed The Algorithms
C++ collection. Tests, build output and vendored material are excluded; exact
duplicates are removed. `provenance.json` records revisions and archive hashes,
and `licenses/` retains the upstream license texts.

### linux_manpages: Linux kernel/userspace interfaces

Linux man-pages 6.19, downloaded from kernel.org with a pinned SHA-256 and
rendered from roff to readable UTF-8 with `groff` and `col`. Redirect-only alias
pages are omitted. This covers system calls, library interfaces, protocols,
file formats, administration interfaces, and `capabilities(7)`. Licensing is
declared per page upstream; all release license texts are retained.

### code_qa and linux_qa: authored instruction examples

Distinct CC0 conversations generated deterministically by
`generate_code_linux_qa.py`; none are repeated in `pretrain_v3`. Code examples
cover Python, Rust, and C++ concepts, implementation, debugging, and
cross-language comparisons. Linux examples cover operations, diagnostics,
shell safety, permissions, services, networking, containers, and the kernel
capability model. Python fenced blocks are parsed during generation.

### code_curriculum and linux_curriculum: large worked-problem sets

`generate_large_qa.py` adds more than ten thousand balanced Python, Rust, and
C++ tracing, implementation, debugging, and testing problems, plus thousands
of Linux filesystem, journal, socket, permissions, shell-safety, and capability
scenarios. Each question/answer pair is unique, deterministic, CC0, and appears
only once in `pretrain_v3`. Numeric outputs are computed by the generator and
Python code is parsed before any records are written.

### tutor_qa: Codex-authored tutor conversations

1,800 conversations (36 batches of 50 across Python, Rust, C++ and Linux topics), each written individually by Codex (`codex exec`, one
headless session per batch). `scripts/tutor_qa/` holds the batch plan
(`batches.json`: 60 topics x 2 levels), the prompt, the runner, and
`validate.py`, which every batch must pass: schema, unique ids and questions,
<= 230 GPT-2 tokens per rendered conversation (context is 256), Python in
answers parses, Rust in answers compiles (`rustc`), C++ in answers compiles
(`g++ -std=c++20 -fsyntax-only`). Buggy code in debugging *questions* is
intentional and unchecked. Batches were generated in a staging directory and
copied here once valid; the remaining 84 planned batches are unrun (Codex usage limit). CC0-1.0.

### python_books: shareable Python books and courses

Seven sources pinned to commits, each with its license statement verified in
the archive and kept in `licenses/`: Practical Python Programming and Advanced
Python Mastery (Beazley, CC BY-SA 4.0), A Byte of Python (CC BY-SA 4.0), Dive
Into Python 3 (CC BY-SA 3.0; HTML converted to text), the Exercism Python track
concept docs and exemplar solutions (MIT), learn-python (MIT), and
TheAlgorithms/Python (MIT). NonCommercial/NoDerivs books (Think Python, py4e,
Hitchhiker's Guide, Python Data Science Handbook text) are deliberately
excluded so the data stays shareable. ~7.5M characters.
