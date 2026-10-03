You are writing training data for a small from-scratch language model that is
learning to be a programming tutor. Write **50 question/answer conversations**
about one topic, by hand, one at a time. This batch:

- Language/domain: {language}
- Topic: {topic}
- Level: {level}
- Output file: batches/{slug}.jsonl   (create it; do not touch any other file in batches/)

## Hard rules

1. **Author every question and every answer individually.** Do NOT write a
   generator script, template, loop, itertools product, or fill-in-the-blank
   pattern to produce records. A previous attempt did that and produced 900
   copies of the same answer paragraph with the nouns swapped; that data is
   useless. You may use Python only to append records you have written to the
   file and to run the validator.
2. **Correctness beats everything.** If you are not sure an answer is right,
   write a different question. Code in answers must be correct and idiomatic
   for {language_name}.
3. **Short.** The whole rendered conversation must be <= 230 GPT-2 tokens
   (the model's context is 256). Most answers should be 30-150 words. Prefer
   one tight code example over several.
4. **No two conversations may be near-duplicates.** Vary the angle, not just
   the variable names.

## Variety (aim for roughly this spread across the 50)

- concept explanation ("what is / why does ...")               ~20%
- how-to / write-the-code ("how do I ...", "write a function that ...") ~25%
- debugging: user pastes buggy code or an error message/traceback, answer
  explains the cause and shows the fix                           ~20%
- comparison / when-to-use-which / best practice                ~15%
- predict the output (answer gives the output and why)          <=10%
- two-turn: user asks, assistant answers, user asks a natural follow-up,
  assistant answers                                             ~10%

Vary the user's voice: terse, beginner-confused, precise, pasted error text.
Match the level: "{level}" means {level_hint}

## Record format (one JSON object per line)

{{"id": "tutor_qa/{slug}_001",
 "messages": [
   {{"role": "system", "content": "You are a patient programming tutor. Give a correct, concise answer, explain the important idea, and use runnable code when it helps."}},
   {{"role": "user", "content": "..."}},
   {{"role": "assistant", "content": "..."}}
 ],
 "metadata": {{"topic": "{topic}", "language": "{language}", "level": "{level}", "kind": "concept|howto|debug|compare|predict|followup", "license": "CC0-1.0", "origin": "synthetic: codex-authored for llm_bobgpt"}}}}

- ids are `tutor_qa/{slug}_001` through `_050`.
- Include the system turn (exactly the text above) in about half of the
  records; omit it in the others (then messages start with the user turn).
- Code goes in fenced blocks tagged with the language: ```python, ```rust,
  ```cpp, ```bash, ```text for output. Rust/C++ in *answers* is compiled
  (Rust as a lib unless it has `fn main`; C++ with -fsyntax-only), so include
  the needed `use`/`#include` lines. Buggy code in *questions* is not checked.

## Finish

Run: `{python} validate.py batches/{slug}.jsonl`
Fix every reported error and re-run until it reports 0 errors. Then reply with
one line: the record count and the kind breakdown.
