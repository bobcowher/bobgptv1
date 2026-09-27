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
python scripts/build_mix.py pretrain_v1          # -> data/build/pretrain_v1/
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
- **Train/val split:** a unit goes to val when `sha256(unit id)` lands in the
  bottom `val_fraction` of 1000 buckets. The split depends only on ids, so
  adding documents never moves existing ones between train and val.
- **Chunking:** documents longer than `chunk_chars` are cut at paragraph
  breaks into `<id>#partNNN` units before hashing. With only 25 books, hashing
  whole books would put 2-3 entire novels in val; hashing ~20K-character parts
  gives val prose from most books.
- Phases (e.g. pretraining vs. mixing in Q&A late) are just different mixes over
  the same sources. `pretrain_v1` doesn't use `python_qa` yet. Adding
  `{"name": "python_qa", "repeat": N}` puts it in the block.

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

### python_qa: synthetic Python Q&A

`python_tutor.jsonl` has 208 deterministic chat examples: concepts, output
prediction, function writing and debugging. The generator checks for
duplicates and parses every fenced Python block with `ast.parse`. Released
under CC0-1.0.
