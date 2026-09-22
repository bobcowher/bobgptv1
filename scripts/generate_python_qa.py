#!/usr/bin/env python3
"""Generate a deterministic, Python-heavy educational Q&A dataset.

The JSONL files use the widely supported ``messages`` schema.  Matching plain
text files are also emitted so this repository's existing next-token trainer
can read them without an instruction-tuning collator.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a patient Python tutor. Give a correct, concise answer, explain the "
    "important idea, and use runnable Python 3 code when it helps."
)


@dataclass(frozen=True)
class Example:
    id: str
    category: str
    question: str
    answer: str


def fenced(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


def number_name(value: int) -> str:
    return f"neg{abs(value)}" if value < 0 else str(value)


def build_examples() -> list[Example]:
    examples: list[Example] = []

    def add(category: str, question: str, answer: str) -> None:
        number = 1 + sum(item.category == category for item in examples)
        examples.append(Example(f"python_{category}_{number:04d}", category, question, answer))

    concepts = [
        ("What is the difference between a list and a tuple in Python?", "A list is mutable: you can add, remove, or replace its items. A tuple is immutable after creation. Both preserve order and may contain mixed types. Use a list for a collection that changes and a tuple for a fixed record or a value that may need to be hashable."),
        ("What does `None` mean in Python?", "`None` is Python's single value for “no value” or “not present.” A function with no explicit `return` returns `None`. Test it with `value is None`, not `value == None`."),
        ("Why should I use `is None` instead of `== None`?", "`is` checks object identity, and `None` is a singleton. `==` can invoke a class's custom equality method and produce surprising results. The idiomatic test is `if value is None:`."),
        ("What is a dictionary comprehension? Give a small example.", "A dictionary comprehension builds a dictionary from an iterable. For example, this maps each number to its square:\n\n" + fenced("squares = {n: n * n for n in range(5)}\nprint(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}")),
        ("Explain the difference between `append` and `extend` on a list.", "`append(x)` adds `x` as one new item. `extend(iterable)` adds each item from the iterable. Starting with `[1]`, `append([2, 3])` produces `[1, [2, 3]]`, while `extend([2, 3])` produces `[1, 2, 3]`."),
        ("What is a Python context manager?", "A context manager performs setup and guaranteed cleanup around a block. The `with` statement uses one. For files, it closes the file even if an exception occurs:\n\n" + fenced("with open('notes.txt', encoding='utf-8') as file:\n    text = file.read()")),
        ("When should a function raise `ValueError`?", "Raise `ValueError` when an argument has the right general type but an unacceptable value, such as a negative radius. Use `TypeError` when the operation or function received an inappropriate type."),
        ("What is the purpose of `enumerate`?", "`enumerate(iterable)` yields `(index, item)` pairs, avoiding a manually managed counter. It also accepts `start`, as in `enumerate(names, start=1)`."),
        ("What is the purpose of `zip`?", "`zip` walks several iterables in parallel and yields tuples. It stops at the shortest input by default. For example, `dict(zip(names, scores))` pairs each name with its score."),
        ("Explain shallow copy versus deep copy.", "A shallow copy creates a new outer container but keeps references to nested objects. A deep copy recursively copies nested objects too. Use `copy.copy` for a shallow copy and `copy.deepcopy` when independent nested state is truly required."),
        ("What does `if __name__ == '__main__':` do?", "Code inside that guard runs when the file is executed as a script, but not when the file is imported as a module. This keeps reusable definitions importable without triggering command-line behavior."),
        ("What is a generator in Python?", "A generator produces values lazily, usually with `yield`, so it need not store the entire result in memory. Iteration resumes after each `yield` until the function returns."),
        ("How is a generator expression different from a list comprehension?", "A list comprehension eagerly creates a list. A generator expression produces one item at a time. Compare `[f(x) for x in xs]` with `(f(x) for x in xs)`. The generator is often more memory-efficient but can normally be consumed only once."),
        ("What is an exception's `finally` block for?", "A `finally` block runs whether the `try` block succeeds or raises an exception. It is meant for cleanup that must happen either way. Prefer a context manager when one naturally represents the resource."),
        ("Why are mutable default arguments risky?", "A default value is evaluated once when the function is defined, so a default list or dictionary is shared across calls. Use `None` and create a fresh object inside:\n\n" + fenced("def collect(item, items=None):\n    if items is None:\n        items = []\n    items.append(item)\n    return items")),
        ("What does `*args` mean in a function definition?", "`*args` collects extra positional arguments into a tuple. It is useful when a function intentionally accepts a variable number of positional values."),
        ("What does `**kwargs` mean in a function definition?", "`**kwargs` collects extra keyword arguments into a dictionary. Explicitly named parameters are clearer when the accepted options are known."),
        ("What is a dataclass useful for?", "A `dataclass` is useful for classes that mainly hold data. The decorator can generate methods such as `__init__`, `__repr__`, and `__eq__`, reducing boilerplate while keeping named attributes and type hints."),
        ("What is the difference between an iterable and an iterator?", "An iterable can produce an iterator, usually through `iter()`. An iterator also has `__next__()` and remembers traversal state. A list is iterable; `iter(a_list)` returns an iterator over it."),
        ("Why use `pathlib.Path` instead of assembling paths with strings?", "`Path` expresses path operations directly, handles platform separators, and provides methods such as `read_text`, `mkdir`, and `glob`. For example: `Path('data') / 'train.jsonl'`."),
        ("What does `sorted(items, key=...)` do?", "It returns a new list ordered by the value produced by the `key` function for each item. The original items are returned, not the keys. Python's sort is stable, so equal-key items keep their relative order."),
        ("What does it mean that Python sorting is stable?", "Items with equal sort keys retain their original relative order. This lets you sort by a secondary key first and then by a primary key, or preserve an earlier meaningful ordering."),
        ("How should I compare floating-point results?", "Do not rely on exact equality for most computed floats. Use `math.isclose(a, b, rel_tol=..., abs_tol=...)`, choosing tolerances appropriate to the problem."),
        ("What is a set good for?", "A set stores unique hashable values and supports fast membership tests and operations such as union, intersection, and difference. Sets do not represent a meaningful positional order."),
        ("What is the difference between `str.find` and `str.index`?", "Both return the first matching position. `find` returns `-1` when the substring is absent, while `index` raises `ValueError`."),
        ("Why use `with open(...)` when reading a file?", "The `with` block closes the file reliably, including when reading raises an exception. Specify an encoding for text files, commonly `encoding='utf-8'`."),
        ("What does `dict.get` do?", "`mapping.get(key, default)` returns the value for `key`, or `default` if the key is absent. Unlike `mapping[key]`, it does not raise `KeyError` for a missing key."),
        ("What is the difference between `break` and `continue`?", "`break` exits the nearest loop completely. `continue` skips the rest of the current iteration and starts the next one."),
        ("What does a leading underscore in `_name` conventionally mean?", "It signals that the name is an internal implementation detail, not part of the public API. It is a convention rather than strong access control."),
        ("What are type hints for?", "Type hints document expected types and enable static analysis, editor assistance, and better API clarity. Python does not enforce ordinary annotations at runtime unless another library does so."),
        ("What is the difference between `@staticmethod` and `@classmethod`?", "A static method receives no automatic instance or class argument. A class method receives the class as `cls`, making it useful for alternate constructors and behavior that should respect subclasses."),
        ("What is a Python slice?", "A slice selects a range using `start:stop:step`; the stop index is excluded. Omitted bounds use the sequence ends, and a negative step can traverse backward. For example, `values[::-1]` returns a reversed copy."),
        ("What is duck typing?", "Duck typing means code relies on supported behavior rather than requiring a specific concrete class. A function may accept any object with a suitable `read()` method, for example."),
        ("What is the difference between `==` and `is`?", "`==` asks whether values compare equal. `is` asks whether two references point to the same object. Use `is` for identity checks such as `value is None`; use `==` for ordinary value comparison."),
        ("Why might `sum(values, [])` be a poor way to flatten lists?", "Repeated list concatenation copies an ever-growing result, leading to quadratic work. A comprehension such as `[item for group in groups for item in group]` is clearer and typically linear."),
        ("What is a pure function?", "A pure function's result depends only on its arguments, and it has no observable side effects. Pure functions are easier to test and reason about, though practical programs also need controlled side effects."),
        ("What does `raise ... from ...` accomplish?", "It explicitly chains exceptions, preserving the original cause while presenting a more meaningful higher-level error. Use `raise NewError(...) from original_error`."),
        ("Why prefer `isinstance(value, SomeClass)` over `type(value) is SomeClass`?", "`isinstance` normally accepts subclasses and can check a tuple of types. Exact `type(...) is ...` checks reject subclass instances and are rarely what an API intends."),
        ("What does `yield from` do?", "`yield from iterable` delegates iteration to another iterable or generator. It is a concise way to forward all values and, for generators, also supports their full send/throw/return protocol."),
        ("What is a closure?", "A closure is a function that retains access to names from its enclosing scope even after that outer function has returned. It can package behavior together with private state."),
    ]
    for question, answer in concepts:
        add("concept", question, answer)

    # Output-prediction exercises: deterministic, short, and easy to verify.
    for start in range(-4, 8):
        stop = start + 6
        step = 2
        values = list(range(start, stop, step))
        code = f"values = list(range({start}, {stop}, {step}))\nprint(values)\nprint(sum(values))"
        add(
            "predict_output",
            f"What does this code print?\n\n{fenced(code)}",
            f"It prints:\n\n```text\n{values}\n{sum(values)}\n```\n\n`range` includes the start and excludes the stop value.",
        )

    words = ["planet", "robot", "python", "nebula", "comet", "signal", "module", "iterator"]
    for word in words:
        for shift in (1, 2, 3):
            code = f"word = {word!r}\nprint(word[{shift}:])\nprint(word[::-1])\nprint(word.count({word[shift]!r}))"
            output = f"{word[shift:]}\n{word[::-1]}\n{word.count(word[shift])}"
            add(
                "predict_output",
                f"Predict the exact output.\n\n{fenced(code)}",
                f"The exact output is:\n\n```text\n{output}\n```\n\nThe first slice drops {shift} leading character(s), and `[::-1]` reverses the string.",
            )

    for limit in range(5, 21):
        divisor = 2 + (limit % 4)
        result = [n * n for n in range(limit) if n % divisor == 0]
        code = f"result = [n * n for n in range({limit}) if n % {divisor} == 0]\nprint(result)"
        add(
            "predict_output",
            f"What is printed by this comprehension?\n\n{fenced(code)}",
            f"It prints `{'{}'.format(result)}`. The filter keeps multiples of {divisor}, then the expression squares them.",
        )

    for count in range(3, 15):
        items = [chr(97 + i) for i in range(count)]
        start = count % 3
        pairs = list(enumerate(items[start:], start=start))
        code = f"letters = {items!r}\nprint(list(enumerate(letters[{start}:], start={start})))"
        add(
            "predict_output",
            f"Predict the output of this `enumerate` example.\n\n{fenced(code)}",
            f"It prints `{pairs}`. The slice and the `start` argument both begin at index {start}.",
        )

    for size in range(3, 13):
        mapping = {n: n**2 for n in range(size)}
        threshold = (size * size) // 3
        result = sorted(key for key, value in mapping.items() if value > threshold)
        code = (
            f"squares = {{n: n ** 2 for n in range({size})}}\n"
            f"result = sorted(k for k, v in squares.items() if v > {threshold})\n"
            "print(result)"
        )
        add(
            "predict_output",
            f"What does this dictionary code print?\n\n{fenced(code)}",
            f"It prints `{result}`. Only keys whose squared value is greater than {threshold} are retained.",
        )

    for left in range(2, 10):
        right = left + 3
        code = f"a = set(range({left}))\nb = set(range({right - 3}, {right}))\nprint(sorted(a & b))\nprint(sorted(a - b))"
        intersection = sorted(set(range(left)) & set(range(right - 3, right)))
        difference = sorted(set(range(left)) - set(range(right - 3, right)))
        add(
            "predict_output",
            f"What is the exact output?\n\n{fenced(code)}",
            f"The output is:\n\n```text\n{intersection}\n{difference}\n```\n\n`&` is intersection and `-` is set difference; sorting makes display order deterministic.",
        )

    implementation_tasks = [
        ("Write `count_vowels(text)` that counts English vowels case-insensitively.", "def count_vowels(text: str) -> int:\n    vowels = set('aeiou')\n    return sum(char.lower() in vowels for char in text)"),
        ("Write `is_palindrome(text)` that ignores case and non-alphanumeric characters.", "def is_palindrome(text: str) -> bool:\n    cleaned = ''.join(char.lower() for char in text if char.isalnum())\n    return cleaned == cleaned[::-1]"),
        ("Write `deduplicate(items)` that preserves the first occurrence order. Assume items are hashable.", "def deduplicate(items):\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result"),
        ("Write `running_totals(numbers)` that returns the cumulative sums.", "def running_totals(numbers):\n    result = []\n    total = 0\n    for number in numbers:\n        total += number\n        result.append(total)\n    return result"),
        ("Write `flatten(groups)` to flatten one level of nested iterables into a list.", "def flatten(groups):\n    return [item for group in groups for item in group]"),
        ("Write `word_frequencies(text)` using whitespace-separated, case-insensitive words.", "def word_frequencies(text: str) -> dict[str, int]:\n    counts: dict[str, int] = {}\n    for word in text.lower().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts"),
        ("Write `transpose(matrix)` for a nonempty rectangular list of lists.", "def transpose(matrix):\n    return [list(column) for column in zip(*matrix)]"),
        ("Write iterative `binary_search(values, target)` for an ascending list. Return the index or `-1`.", "def binary_search(values, target):\n    low, high = 0, len(values) - 1\n    while low <= high:\n        middle = (low + high) // 2\n        if values[middle] == target:\n            return middle\n        if values[middle] < target:\n            low = middle + 1\n        else:\n            high = middle - 1\n    return -1"),
        ("Write `chunks(items, size)` that returns consecutive list chunks and rejects nonpositive sizes.", "def chunks(items, size):\n    if size <= 0:\n        raise ValueError('size must be positive')\n    return [items[index:index + size] for index in range(0, len(items), size)]"),
        ("Write `merge_counts(left, right)` that adds values for shared dictionary keys without mutating either input.", "def merge_counts(left, right):\n    merged = dict(left)\n    for key, value in right.items():\n        merged[key] = merged.get(key, 0) + value\n    return merged"),
        ("Write `safe_divide(numerator, denominator)` that returns `None` when the denominator is zero.", "def safe_divide(numerator, denominator):\n    if denominator == 0:\n        return None\n    return numerator / denominator"),
        ("Write `parse_ints(text)` that parses comma-separated integers and tolerates surrounding spaces.", "def parse_ints(text: str) -> list[int]:\n    if not text.strip():\n        return []\n    return [int(part.strip()) for part in text.split(',')]"),
        ("Write `group_by_first_letter(words)` with lowercase first letters as keys. Skip empty strings.", "def group_by_first_letter(words):\n    groups = {}\n    for word in words:\n        if word:\n            groups.setdefault(word[0].lower(), []).append(word)\n    return groups"),
        ("Write `balanced_parentheses(text)` for strings containing `(` and `)` plus arbitrary other characters.", "def balanced_parentheses(text: str) -> bool:\n    depth = 0\n    for char in text:\n        if char == '(':\n            depth += 1\n        elif char == ')':\n            depth -= 1\n            if depth < 0:\n                return False\n    return depth == 0"),
        ("Write `common_items(left, right)` returning unique shared hashable items in their first-seen order from `left`.", "def common_items(left, right):\n    right_items = set(right)\n    seen = set()\n    result = []\n    for item in left:\n        if item in right_items and item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result"),
        ("Write `invert_unique(mapping)` that swaps keys and values. Raise `ValueError` if values are not unique.", "def invert_unique(mapping):\n    inverted = {}\n    for key, value in mapping.items():\n        if value in inverted:\n            raise ValueError('values must be unique')\n        inverted[value] = key\n    return inverted"),
        ("Write `second_largest(numbers)` using distinct values, raising `ValueError` if fewer than two exist.", "def second_largest(numbers):\n    unique = set(numbers)\n    if len(unique) < 2:\n        raise ValueError('need at least two distinct values')\n    largest = max(unique)\n    unique.remove(largest)\n    return max(unique)"),
        ("Write `normalize_spaces(text)` that collapses all whitespace runs to one space and trims the ends.", "def normalize_spaces(text: str) -> str:\n    return ' '.join(text.split())"),
        ("Write `rotate_left(items, amount)` without mutating the input. An empty input should return `[]`.", "def rotate_left(items, amount):\n    if not items:\n        return []\n    offset = amount % len(items)\n    return list(items[offset:]) + list(items[:offset])"),
        ("Write `matrix_diagonal(matrix)` returning the main diagonal of a square matrix.", "def matrix_diagonal(matrix):\n    size = len(matrix)\n    if any(len(row) != size for row in matrix):\n        raise ValueError('matrix must be square')\n    return [matrix[index][index] for index in range(size)]"),
        ("Write `longest_word(words)` returning the first longest word, or `None` for an empty iterable.", "def longest_word(words):\n    iterator = iter(words)\n    try:\n        longest = next(iterator)\n    except StopIteration:\n        return None\n    for word in iterator:\n        if len(word) > len(longest):\n            longest = word\n    return longest"),
        ("Write `partition(items, predicate)` returning `(matches, non_matches)` while preserving order.", "def partition(items, predicate):\n    matches = []\n    non_matches = []\n    for item in items:\n        (matches if predicate(item) else non_matches).append(item)\n    return matches, non_matches"),
        ("Write `read_nonempty_lines(path)` using UTF-8. Strip whitespace and omit blank lines.", "from pathlib import Path\n\ndef read_nonempty_lines(path):\n    text = Path(path).read_text(encoding='utf-8')\n    return [line.strip() for line in text.splitlines() if line.strip()]"),
        ("Write `mean(numbers)` that raises `ValueError` for an empty iterable and also works with generators.", "def mean(numbers):\n    total = 0\n    count = 0\n    for number in numbers:\n        total += number\n        count += 1\n    if count == 0:\n        raise ValueError('mean requires at least one value')\n    return total / count"),
        ("Write recursive `factorial(n)` that rejects negative integers.", "def factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('n must be nonnegative')\n    if n < 2:\n        return 1\n    return n * factorial(n - 1)"),
        ("Write `fibonacci(n)` returning the first `n` Fibonacci numbers, starting with 0 and 1.", "def fibonacci(n: int) -> list[int]:\n    if n < 0:\n        raise ValueError('n must be nonnegative')\n    result = []\n    a, b = 0, 1\n    for _ in range(n):\n        result.append(a)\n        a, b = b, a + b\n    return result"),
        ("Write `compose(f, g)` so the returned function computes `f(g(x))`.", "def compose(f, g):\n    def composed(value):\n        return f(g(value))\n    return composed"),
        ("Write `get_nested(mapping, keys, default=None)` to follow a sequence of dictionary keys safely.", "def get_nested(mapping, keys, default=None):\n    current = mapping\n    for key in keys:\n        if not isinstance(current, dict) or key not in current:\n            return default\n        current = current[key]\n    return current"),
        ("Write `pairwise(items)` returning adjacent pairs such as `[(1, 2), (2, 3)]`.", "def pairwise(items):\n    return list(zip(items, items[1:]))"),
        ("Write `title_counts(records)` counting the `title` field in a sequence of dictionaries; missing titles count as `'<missing>'`.", "def title_counts(records):\n    counts = {}\n    for record in records:\n        title = record.get('title', '<missing>')\n        counts[title] = counts.get(title, 0) + 1\n    return counts"),
    ]
    for prompt, code in implementation_tasks:
        add("write_code", prompt, "Here is a direct implementation:\n\n" + fenced(code))

    # Parameterized implementation exercises reinforce core control flow while
    # varying names, constants, and requirements.
    for threshold in range(-5, 15):
        function_name = f"values_above_{number_name(threshold)}"
        code = f"def {function_name}(numbers):\n    return [number for number in numbers if number > {threshold}]"
        add(
            "write_code",
            f"Write `{function_name}(numbers)` that returns all values strictly greater than {threshold}, preserving order.",
            "A list comprehension expresses the filter directly:\n\n" + fenced(code),
        )

    for divisor in range(2, 13):
        code = (
            f"def count_multiples_of_{divisor}(numbers):\n"
            f"    return sum(number % {divisor} == 0 for number in numbers)"
        )
        add(
            "write_code",
            f"Write `count_multiples_of_{divisor}(numbers)` that counts integers divisible by {divisor}.",
            "Booleans sum as 1 for true and 0 for false:\n\n" + fenced(code),
        )

    for low in range(-5, 5):
        high = low + 10
        function_name = f"clamp_{number_name(low)}_{number_name(high)}"
        code = f"def {function_name}(value):\n    return min({high}, max({low}, value))"
        add(
            "write_code",
            f"Write `{function_name}(value)` that limits a number to the inclusive range [{low}, {high}].",
            "Clamp against the lower bound first, then the upper bound:\n\n" + fenced(code),
        )

    debugging = [
        ("This function unexpectedly remembers earlier calls. Fix it.\n\n" + fenced("def add_item(item, items=[]):\n    items.append(item)\n    return items"), "The default list is created once and shared. Use `None` as a sentinel:\n\n" + fenced("def add_item(item, items=None):\n    if items is None:\n        items = []\n    items.append(item)\n    return items")),
        ("Fix this off-by-one error so `countdown(3)` returns `[3, 2, 1, 0]`.\n\n" + fenced("def countdown(n):\n    return list(range(n, 0, -1))"), "`range` excludes its stop value, so stop at `-1`:\n\n" + fenced("def countdown(n):\n    return list(range(n, -1, -1))")),
        ("Fix this aliasing bug so changing `copy` does not change the nested list in `original`.\n\n" + fenced("original = [[1], [2]]\ncopy = original.copy()\ncopy[0].append(9)"), "A shallow copy shares nested lists. Deep-copy the nested structure:\n\n" + fenced("import copy\n\noriginal = [[1], [2]]\nindependent = copy.deepcopy(original)\nindependent[0].append(9)")),
        ("This loop should remove every even number, but it skips values. Fix it.\n\n" + fenced("numbers = [2, 4, 6, 7]\nfor number in numbers:\n    if number % 2 == 0:\n        numbers.remove(number)"), "Do not mutate a list while iterating over it. Build the desired result:\n\n" + fenced("numbers = [2, 4, 6, 7]\nnumbers = [number for number in numbers if number % 2 != 0]")),
        ("Fix the comparison bug.\n\n" + fenced("def is_ten(value):\n    return value is 10"), "`is` checks identity; numeric values should be compared with `==`:\n\n" + fenced("def is_ten(value):\n    return value == 10")),
        ("Fix this function so it actually returns the doubled values.\n\n" + fenced("def doubled(values):\n    result = [value * 2 for value in values]"), "Return the computed list:\n\n" + fenced("def doubled(values):\n    result = [value * 2 for value in values]\n    return result")),
        ("Fix this file-reading function so the returned file object is not already closed. Return its text instead.\n\n" + fenced("def read_file(path):\n    with open(path, encoding='utf-8') as file:\n        return file"), "Read inside the context manager and return the resulting string:\n\n" + fenced("def read_file(path):\n    with open(path, encoding='utf-8') as file:\n        return file.read()")),
        ("Fix the missing base case in this recursive function.\n\n" + fenced("def total(values):\n    return values[0] + total(values[1:])"), "Return zero for an empty sequence:\n\n" + fenced("def total(values):\n    if not values:\n        return 0\n    return values[0] + total(values[1:])")),
        ("This exception handler hides every kind of failure. Narrow it to invalid integer input.\n\n" + fenced("try:\n    count = int(text)\nexcept:\n    count = 0"), "Catch the documented conversion error, not `BaseException` subclasses such as `KeyboardInterrupt`:\n\n" + fenced("try:\n    count = int(text)\nexcept ValueError:\n    count = 0")),
        ("Fix this method definition.\n\n" + fenced("class Counter:\n    def increment(amount=1):\n        self.value += amount"), "Instance methods receive `self` first:\n\n" + fenced("class Counter:\n    def increment(self, amount=1):\n        self.value += amount")),
        ("Fix this boolean condition so it accepts only `'yes'` or `'y'`.\n\n" + fenced("if answer == 'yes' or 'y':\n    accepted = True"), "Each alternative must be compared, or use membership:\n\n" + fenced("if answer in {'yes', 'y'}:\n    accepted = True")),
        ("Fix the function so it keeps zero values instead of replacing them with 10.\n\n" + fenced("def normalize(value):\n    return value or 10"), "Test specifically for `None`, because zero is a valid falsey value:\n\n" + fenced("def normalize(value):\n    return 10 if value is None else value")),
        ("Fix this sorting call so names are ordered case-insensitively.\n\n" + fenced("names = ['zoe', 'Amy', 'bob']\nnames.sort(str.lower())"), "`key` is a keyword-only argument for `list.sort`:\n\n" + fenced("names = ['zoe', 'Amy', 'bob']\nnames.sort(key=str.lower)")),
        ("Fix this dictionary loop so both keys and values are unpacked.\n\n" + fenced("for key, value in scores:\n    print(key, value)"), "Iterating over a dictionary alone yields keys. Iterate over its items:\n\n" + fenced("for key, value in scores.items():\n    print(key, value)")),
        ("Fix this class constructor typo.\n\n" + fenced("class Point:\n    def _init_(self, x, y):\n        self.x = x\n        self.y = y"), "Python's constructor hook uses two underscores on each side:\n\n" + fenced("class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y")),
    ]
    for question, answer in debugging:
        add("debug", question, answer)

    return examples


def validate(examples: list[Example]) -> None:
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("duplicate example IDs")
    pairs = [(example.question, example.answer) for example in examples]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate question/answer pairs")
    for example in examples:
        if not example.question.strip() or not example.answer.strip():
            raise ValueError(f"empty content in {example.id}")
        for code in re.findall(r"```python\n(.*?)```", example.question + example.answer, re.DOTALL):
            try:
                ast.parse(code)
            except SyntaxError as error:
                raise ValueError(f"invalid Python in {example.id}: {error}") from error


def split_name(example_id: str) -> str:
    # A stable content-independent 90/10 split.
    bucket = int(hashlib.sha256(example_id.encode()).hexdigest()[:8], 16) % 10
    return "validation" if bucket == 0 else "train"


def as_record(example: Example) -> dict:
    return {
        "id": example.id,
        "category": example.category,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example.question},
            {"role": "assistant", "content": example.answer},
        ],
        "license": "CC0-1.0",
        "source": "synthetic: authored for llm_bobgpt",
    }


def as_text(example: Example) -> str:
    return (
        f"### System\n{SYSTEM_PROMPT}\n\n"
        f"### Question\n{example.question.strip()}\n\n"
        f"### Answer\n{example.answer.strip()}\n\n"
        "### End"
    )


def write_dataset(output_dir: Path) -> None:
    examples = build_examples()
    validate(examples)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        name: [example for example in examples if split_name(example.id) == name]
        for name in ("train", "validation")
    }

    for name, items in splits.items():
        jsonl = "\n".join(json.dumps(as_record(item), ensure_ascii=False) for item in items) + "\n"
        text = "\n\n\n".join(as_text(item) for item in items) + "\n"
        (output_dir / f"{name}.jsonl").write_text(jsonl, encoding="utf-8")
        (output_dir / f"{name}.txt").write_text(text, encoding="utf-8")

    categories = Counter(example.category for example in examples)
    metadata = {
        "description": "Synthetic Python question/answer examples for educational instruction tuning.",
        "license": "CC0-1.0",
        "generator": "scripts/generate_python_qa.py",
        "format": "JSON Lines with system/user/assistant messages; plain-text mirrors are also provided.",
        "total_examples": len(examples),
        "split_counts": {name: len(items) for name, items in splits.items()},
        "category_counts": dict(sorted(categories.items())),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "finetune",
    )
    args = parser.parse_args()
    write_dataset(args.output_dir)


if __name__ == "__main__":
    main()
