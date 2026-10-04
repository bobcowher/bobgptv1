# Evaluating a run

How we decide whether a training run made bobgpt better, which numbers can be
compared with which, and the commands that produce them. Results so far are in
the table at the end.

There are four instruments. Each answers a different question, and each has a
blind spot, so a run is judged on all of them together:

| # | Instrument | Question it answers | Comparable across |
|---|---|---|---|
| 1 | Training log val loss | Is this run still learning? | Steps of the *same* run only |
| 2 | `scripts/eval_frozen.py` | Did pretraining get better at text and Python docs? | Every run since 20 |
| 3 | Post-training masked val | Did post-training get better at answering? | Runs on the same post-train mix |
| 4 | `scripts/test.py` samples | What does it actually say? | Anything, by eye |

## 1. Training log val loss

Every `eval_freq` steps the trainer prints a line like this (run 28, the
final step):

```
Ep 1 (Step 158000): Train loss 2.746, Val loss 2.479, LR 6.00e-05, 30,346 tok/s
Ep 1 done: Epoch train loss 3.247, Epoch val loss 2.428
```

The per-step `Val loss` reads only `eval_iter` (20) random batches, so it jumps
around by ±0.1 between evals. The `Epoch val loss` reads the whole val split and
is the number checkpointing and early stopping use.

Use it to watch a run: is loss still falling, has it plateaued, is val rising
while train falls (overfitting). **Don't** compare it between runs on different
mixes. The val split is built from the mix, so when the mix changes, so does
what val measures. Run 28's 2.428 looks worse than run 25's 2.058, but v10's
val is mostly web text and GitHub code, while v7's had far more Python docs and
Q&A, which are easier to predict. They're different tests.

Throughput (`tok/s`) is logged here too, and in TensorBoard as
`throughput/tokens_per_sec`.

## 2. The frozen benchmark: `scripts/eval_frozen.py`

The fixed yardstick for pretraining. It scores held-out documents from two
sources, `books` and the original nine projects of `python_docs`, that no mix
has trained on since run 20. Units are picked by the same id hash
`build_mix.py` uses for val (`val_fraction` 0.1, `chunk_chars` 20000), so as
long as a mix keeps those two sources at the 0.1 default, they stay held out.
Check that before trusting a new mix's score:

```bash
python3 -c "import json; m=json.load(open('mixes/pretrain_v10.json')); print(m['val_fraction'], m['chunk_chars'], [s for s in m['sources'] if s['name'] in ('books','python_docs')])"
```
```
0.1 20000 [{'name': 'python_docs'}, {'name': 'books'}]
```

(No per-source `val_fraction` override, so both use 0.1.)

Run it on any number of checkpoints. It builds each model at the context
length stored in its weights:

```bash
python scripts/eval_frozen.py checkpoints/run25_model.pth checkpoints/run28_model.pth
EVAL_CTX=1024 python scripts/eval_frozen.py checkpoints/run25_model.pth checkpoints/run28_model.pth
```
```
books: 240,488 tokens  python_docs: 1,527,600 tokens
checkpoints/run25_model.pth  books 3.885  python_docs 1.892
checkpoints/run28_model.pth  books 3.902  python_docs 1.902
```

The numbers are mean loss per token, lower is better. A difference of 0.01 is
real here (1.7M tokens, no sampling), but small.

**Window length.** By default it scores in 256-token windows, because runs
before 24 had a 256 context and this keeps every run on equal terms.
`EVAL_CTX=1024` scores in 1024 windows, which is how runs 24 onward actually
work. Don't mix the two columns:

| | books @256 | python_docs @256 | books @1024 | python_docs @1024 |
|---|---|---|---|---|
| run 25 | 3.885 | 1.892 | 3.767 | 1.706 |
| run 28 | 3.902 | 1.902 | **3.751** | **1.687** |

Run 28 is slightly worse at 256 and slightly better at 1024. Its v10 data is
longer documents (whole GitHub files, web pages), so it learned to use long
context more and short windows a bit less.

**Blind spots.** This benchmark only knows books and Python docs:

- It can't see code, C++, Rust, shell, or general web text. Run 28 spent most
  of its 1.3B tokens on exactly those (v10 is ~25% GitHub code, ~70% FineWeb), and the
  benchmark scores it as "about the same as run 25". That's a limit of the
  yardstick, not proof the extra data did nothing. A second frozen set with
  held-out GitHub and FineWeb documents would fix it.
- Post-training makes these numbers worse on purpose (next section), so only
  compare pretrained checkpoints with each other here.

Runs 19 and earlier used an older split and trained on these documents; their
scores are invalid.

## 3. Post-training: masked val loss

Post-training (`scripts/posttrain.py`, mix `posttrain_v1`) only counts loss on
assistant replies, using the `*_mask.bin` files `build_mix.py` writes for mixes
with `"loss_mask": true`. Its val split is the held-out chat and Q&A records
(no replay text), so the number means "how well does it predict a good answer,
given the question".

The script prints the starting model's score on that val set before it trains
anything, and that's the baseline to beat:

```
Val loss before post-training: 2.813
...
Ep 1 done: Epoch train loss 2.649, Epoch val loss 2.439
Ep 2 done: Epoch train loss 2.300, Epoch val loss 2.396
```

(Run 30: run 28's weights through `posttrain_v1`.)

This is the instrument for comparing pretraining recipes *by what they're for*:
put two pretrained models through the identical post-training and see which
answers better. That's how we tested the pretrain/post-train split:

| | Pretrained on | Before | After | Change |
|---|---|---|---|---|
| run 29 | run 25 (v7: web + docs + Q&A mixed in) | 2.648 | 2.543 (3 epochs) | -0.105 |
| run 30 | run 28 (v10: web + code, no Q&A) | 2.813 | **2.396** (2 epochs) | -0.417 |

Run 28 starts worse (it has never seen the `### Question` format) and finishes
0.15 better.

**Rules for comparing:**

- Same post-train mix only. If a source in it grows (e.g. tutor_qa round 2 is
  installed), new records join val and the number shifts. Bump the mix name
  (`posttrain_v2`) when that happens, rather than rebuilding `posttrain_v1`.
- Compare final epoch val, not per-step lines (same ±0.1 noise as section 1).
- The cost shows up in section 2: post-training worsens the frozen benchmark
  (run 25 → 29: books 3.885 → 4.060 @256; run 28 → 30: 3.902 → 4.049). The
  model gives up some raw text prediction to answer questions. The 11% replay
  in `posttrain_v1` limits that; more replay is the knob if it gets worse.

On run 25, the third epoch only moved val 2.548 → 2.543 while train loss kept
falling, so `posttrain.py` now runs 2.

## 4. Samples: `scripts/test.py`

Loss says nothing about whether answers are right. `test.py` prints
completions for fixed prompts: prose, code starts in Python/Rust/C++, Q&A, and
chat. It's seeded, so two checkpoints get the same random draws:

```bash
python scripts/test.py checkpoints/run30_posttrain_model.pth
```

Run 30, two of the answers:

```
### Question
What does the `yield` keyword do?

### Answer
It returns a generator object, so each `yield` succeeds.
[code block omitted]

### Question
thanks, that helped

### Answer
You’re glad I could help

### End
```

What to look for, roughly in the order the model gets them right:

1. Format: answers start right away and end with `### End`.
2. Register: chat prompts get short chat replies, technical ones get code.
3. Code that looks like the language (balanced braces, real APIs).
4. Correct facts. Still mostly missing at 124M: the list/tuple answer contradicts
   itself in both runs 29 and 30.

One sample per prompt is noisy; don't call a winner from one answer. For a
closer look, chat with it through the API in Open WebUI (`docs/SERVING.md`).

## After a run: the checklist

After a **pretraining** run:

1. Read the end of the log: did it finish (`status : COMPLETED`), and did the
   loss curve flatten or was it still falling?
2. Copy the checkpoint aside on lab so the next run can't overwrite it:
   `cp data/checkpoints/pretrain/model.pth data/checkpoints/run<N>/model.pth`
   (both under `/data/datasets/bobgptv1/` on lab).
3. Pull it: `scp lab:/data/datasets/bobgptv1/checkpoints/run<N>/model.pth checkpoints/run<N>_model.pth`.
   `download_models.sh` doesn't work for runs 28 and later: it fetches
   Beekeeper's per-run `checkpoints/` output, and checkpoints now go to
   `data/checkpoints/` instead.
4. `eval_frozen.py` at 256 and 1024 against the previous best.
5. Post-train it (`INIT_CHECKPOINT=data/checkpoints/run<N>/model.pth` on the
   `bobgpt-v1-posttrain` Beekeeper project) and compare masked val with the
   table above.
6. `test.py` on the post-trained model.
7. Add a row to the results table.

After a **post-training** run: steps 2–3 for `data/checkpoints/posttrain/`,
then 6, then section 2 to see how much it forgot.

## Results

Frozen benchmark at 256 windows unless marked. "Epoch val" is the mix's own val
(section 1), not comparable across mixes.

| Run | Data | Epochs | Epoch val | books | python_docs | Masked val (post-train) |
|---|---|---|---|---|---|---|
| 20 | pretrain_v1 (15M tok) | 10 | 2.200 | 4.171 | 1.890 | |
| 21 | v2 (+FineWeb 200M), constant LR | 5 | 3.326 | 4.122 | 2.270 | |
| 22 | v4, cosine LR | 1 | 3.053 | 3.933 | 2.085 | |
| 23 | v5 (python_docs 50M) | 2 | 2.205 | 3.817 | 1.838 | |
| 24 | v6, context 1024 | 2 | 2.091 | 3.909 (3.795 @1024) | 1.913 (1.728 @1024) | |
| 25 | v7 (+chat, oasst2, smoltalk), LR 6e-4 | 2 | 2.058 | 3.885 (3.767 @1024) | 1.892 (1.706 @1024) | 2.648 (untuned) |
| 28 | v10 pretrain only (1.3B tok, web + code) | 1 | 2.428 | 3.902 (3.751 @1024) | 1.902 (1.687 @1024) | 2.813 (untuned) |
| 29 | run 25 + posttrain_v1 | 3 | | 4.060 (3.921 @1024) | 1.985 (1.780 @1024) | 2.543 |
| 30 | run 28 + posttrain_v1 | 2 | | 4.049 (3.889 @1024) | 2.005 (1.773 @1024) | **2.396** |

Runs 26 and 27 were a speed benchmark and a crash.
