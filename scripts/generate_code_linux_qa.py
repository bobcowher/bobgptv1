#!/usr/bin/env python3
"""Generate varied code and Linux instructional conversations.

The examples are deterministic and authored for this project.  Code material
covers Python, Rust, and modern C++; Linux material covers everyday operations,
diagnostics, shell safety, permissions, services, networking, containers, and
the kernel capability model.  Nothing is multiplied here or in the mix: every
record is a distinct teaching example.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    dataset: str
    category: str
    topic: str
    question: str
    answer: str
    language: str | None = None
    difficulty: str = "intermediate"


SYSTEM_PROMPTS = {
    "code": (
        "You are a careful programming tutor. Give a correct, practical answer, "
        "explain the reasoning, mention important edge cases, and prefer clear, "
        "idiomatic code over clever code."
    ),
    "linux": (
        "You are a careful Linux systems tutor. Give commands that can be understood "
        "before they are run, distinguish observation from mutation, explain privilege "
        "requirements, and call out destructive or security-sensitive operations."
    ),
}


def fenced(language: str, code: str) -> str:
    return f"```{language}\n{code.rstrip()}\n```"


def concepts() -> list[Example]:
    rows: dict[str, list[tuple[str, str, str]]] = {
        "python": [
            ("iteration", "Why is iterating directly over a collection usually better than indexing it with `range(len(...))`?", "Direct iteration expresses intent, works with any iterable, and avoids indexing mistakes. Use `enumerate(items)` when both the position and value are needed. Indexing is appropriate when neighboring positions or in-place replacement are genuinely part of the algorithm."),
            ("generators", "When should a Python function use `yield` instead of returning a list?", "Use `yield` when values can be produced incrementally, the input may be large or unbounded, or the caller might stop early. A generator trades random access and reuse for low memory use and lazy work. Return a list when the complete materialized result is small and callers need indexing or repeated traversal."),
            ("exceptions", "What is the practical difference between `raise` and `raise ... from ...` in Python?", "Bare `raise` rethrows the active exception unchanged. `raise NewError(...) from error` translates it while preserving an explicit causal chain. Use `from` at abstraction boundaries so callers receive a domain-level error without losing the original failure. `from None` intentionally suppresses a noisy context."),
            ("resources", "Why is a context manager useful beyond opening files?", "A context manager makes acquisition and cleanup one structured operation. Locks, database transactions, temporary directories, decimal contexts, and redirected output all benefit because `__exit__` runs when the block exits normally or through an exception. It prevents cleanup logic from being scattered across return and error paths."),
            ("typing", "What do Python type hints guarantee at runtime?", "Ordinary annotations are metadata; Python does not generally enforce them at runtime. Static checkers, editors, documentation tools, and frameworks may interpret them. Runtime validation requires explicit code or a library. Type hints are still valuable because they describe interfaces and let tools catch inconsistent uses before execution."),
            ("dataclasses", "When is a `dataclass` a better fit than a dictionary?", "A dataclass is useful when a record has a stable schema, named fields, and behavior or invariants. It provides generated initialization, representation, and equality while remaining a normal class. A dictionary is better for truly dynamic keys, loose JSON-like data, or schemas not known until runtime."),
            ("mutability", "Explain the bug caused by a mutable default argument and the idiomatic fix.", "Default arguments are evaluated once when the function is defined. A default list or dictionary is therefore shared across calls. Use `None` as a sentinel and allocate inside the function. An immutable default such as a string, integer, or tuple is safe."),
            ("concurrency", "How do threads, processes, and `asyncio` differ for Python workloads?", "Threads are convenient for blocking I/O but the usual CPython GIL limits parallel Python bytecode. Processes provide CPU parallelism at the cost of serialization and separate memory. `asyncio` multiplexes many cooperative I/O tasks in one thread and works best when the libraries in the call chain are async-aware."),
            ("collections", "When would you choose `collections.deque` over a list?", "A deque supports efficient append and pop operations at both ends. A list is efficient at the right end but inserting or removing at the front shifts all later elements. Use a deque for queues, bounded histories, and breadth-first search; use a list when random indexing is important."),
            ("interfaces", "What problem does a Python protocol solve?", "A protocol describes required behavior structurally: an object matches if it has the specified methods or attributes, without inheriting a particular base class. This formalizes duck typing for static analysis and keeps consumers coupled to a small capability rather than a concrete implementation."),
            ("closures", "What does `nonlocal` do in a Python closure?", "`nonlocal name` makes assignments target the nearest enclosing function scope rather than creating a new local variable. It does not refer to globals. It is useful for small stateful closures, though a class is often clearer once the state or behavior grows."),
            ("descriptors", "How are functions on a Python class related to descriptors?", "Functions implement the descriptor protocol. Accessing one through an instance calls its `__get__`, producing a bound method that supplies the instance as the first argument. `staticmethod`, `classmethod`, and `property` are other descriptors that customize class attribute access."),
            ("imports", "Why should import-time side effects be kept small?", "Imports may happen during startup, test discovery, worker creation, tooling, or plugin inspection. Network calls, file mutations, and expensive computation make those contexts slow and surprising. Define functions and classes at module scope, then put executable application work behind a function and an `if __name__ == '__main__':` guard."),
            ("testing", "What makes a unit test resilient rather than brittle?", "A resilient test asserts externally meaningful behavior, controls nondeterminism, and uses the smallest realistic boundary. It avoids depending on private implementation order, incidental formatting, or excessive mocks. Clear arrange-act-assert structure and focused failure messages make regressions easier to diagnose."),
            ("data-model", "What is the difference between `__repr__` and `__str__`?", "`__repr__` is the unambiguous developer-facing representation used by containers and the interactive prompt; when practical, it should resemble valid construction syntax. `__str__` is the friendly user-facing form used by `str()` and `print()`. If `__str__` is absent, Python falls back to `__repr__`."),
        ],
        "rust": [
            ("ownership", "What does Rust ownership prevent that garbage collection alone does not?", "Ownership and borrowing prevent use-after-free, double free, invalidated references, and many data races at compile time. Garbage collection mainly determines when unreachable memory is reclaimed; it does not by itself prevent unsynchronized mutation or references into moved containers. Rust obtains these guarantees without a tracing runtime."),
            ("borrowing", "Why can Rust have either many immutable references or one mutable reference to a value?", "The rule prevents mutation while aliases may be observing the value. Many readers are safe because none can change it; one exclusive writer is safe because no competing reference can read or write simultaneously. The compiler checks the relevant lifetimes, often ending a borrow at its last use."),
            ("lifetimes", "What does a lifetime annotation mean on a Rust function signature?", "It names a relationship among reference lifetimes; it does not extend how long any value lives. For example, tying an output reference to two input references says the output cannot outlive either applicable input. The borrow checker uses that contract at call sites."),
            ("option-result", "When should a Rust API return `Option<T>` versus `Result<T, E>`?", "Return `Option<T>` when absence is an expected state and there is no useful failure explanation, such as a map lookup. Return `Result<T, E>` when the caller needs to distinguish success from a reason for failure. Avoid encoding recoverable failure with panics."),
            ("question-operator", "What exactly does Rust's `?` operator do?", "On success it unwraps the value and execution continues. On failure it returns early from the enclosing function after converting the error through `From` when necessary. It works with compatible residual types such as `Result` and `Option`; it is concise propagation, not exception handling."),
            ("traits", "How is a Rust trait different from a class inheritance hierarchy?", "A trait defines shared behavior that types may implement, including implementations added separately when coherence rules allow. Types do not inherit storage or implementation identity from a base class. Composition, trait bounds, generics, and trait objects cover different forms of reuse and polymorphism."),
            ("generics", "What is monomorphization in Rust?", "For ordinary generic code, the compiler generates specialized machine code for the concrete types actually used. This enables static dispatch and optimization with no per-call virtual lookup, but many instantiations can increase compile time and binary size. A `dyn Trait` uses dynamic dispatch instead."),
            ("smart-pointers", "Compare `Box<T>`, `Rc<T>`, and `Arc<T>`.", "`Box<T>` gives single ownership with heap allocation. `Rc<T>` gives non-thread-safe shared ownership through reference counting. `Arc<T>` uses atomic reference counts for cross-thread sharing. Neither `Rc` nor `Arc` permits arbitrary mutation; combine with `RefCell`, `Mutex`, or `RwLock` according to the access model."),
            ("interior-mutability", "Why does Rust have `RefCell<T>` if borrowing is checked at compile time?", "Some safe designs cannot be proven from local static information. `RefCell<T>` moves the shared-or-exclusive borrow rule to runtime for single-threaded code and panics on violation. It enables interior mutability behind an immutable outer reference, but should not be used to evade a design that can be expressed statically."),
            ("iterators", "Why are Rust iterator chains often as fast as explicit loops?", "Iterator adapters are generic and lazy. Monomorphization and inlining expose the full chain to optimization, so abstraction layers commonly disappear and no intermediate collection is created unless requested. Performance should still be measured when bounds checks, allocation, or complex closures matter."),
            ("strings", "Why are Rust strings not indexable by integer?", "A `String` is UTF-8, and a byte offset may fall inside a code point while a human-perceived character may span several code points. Integer indexing would have unclear semantics and misleading complexity. Use bytes, `chars()`, grapheme-aware libraries, or validated string slices depending on the intended unit."),
            ("moves", "What happens when a `String` is assigned to another Rust variable?", "Ownership moves: the pointer, length, and capacity are copied, and the old binding can no longer be used. The heap buffer is not deep-copied. Call `clone()` only when an independent buffer is actually needed, or borrow with `&str` when temporary access is sufficient."),
            ("concurrency", "What does `Send` versus `Sync` mean in Rust?", "A type is `Send` if ownership can safely cross a thread boundary. It is `Sync` if shared references to it can safely be used from multiple threads; equivalently, `&T` is `Send`. These are unsafe marker traits usually derived automatically from a type's fields."),
            ("macros", "When is a Rust macro preferable to a generic function?", "A macro is appropriate when syntax must vary, the number or kinds of arguments are not expressible as a normal function, code must be generated, or identifiers/items must be produced. Prefer functions and generics for ordinary value-level abstraction because they are simpler to type-check, document, and debug."),
            ("unsafe", "What guarantee must an `unsafe` Rust block uphold?", "`unsafe` permits a small set of operations the compiler cannot verify; it does not disable the borrow checker. The author must uphold every safety invariant required by the operation, such as pointer validity, alignment, initialization, aliasing, and thread safety. A safe wrapper must prevent callers from violating those invariants."),
        ],
        "cpp": [
            ("raii", "What is RAII in C++, and why is it useful for more than memory?", "Resource Acquisition Is Initialization ties a resource's lifetime to an object's lifetime. Constructors establish ownership and destructors release it during normal return or stack unwinding. The pattern manages locks, files, sockets, transactions, and temporary state as well as memory."),
            ("smart-pointers", "When should C++ code use `unique_ptr`, `shared_ptr`, or a plain reference?", "Use `unique_ptr` for exclusive dynamic ownership and transfer it with a move. Use `shared_ptr` only for genuine shared lifetime, while planning how cycles are broken with `weak_ptr`. Use a reference or pointer for non-owning access whose lifetime is guaranteed elsewhere. Prefer direct values when allocation is unnecessary."),
            ("rule-of-zero", "What does the C++ rule of zero recommend?", "Put ownership in members that already manage themselves, such as containers, strings, and smart pointers, so the compiler-generated destructor, copy, and move operations are correct. Hand-written special members are then unnecessary. This is safer than manually coordinating raw resources across five operations."),
            ("value-categories", "Why does `std::move` not actually move anything by itself?", "`std::move` is a cast that marks an expression as eligible to have its resources transferred. A move constructor or move assignment operator performs the transfer. If the destination operation has no move overload—or the source is const—a copy may still occur."),
            ("views", "What lifetime hazard comes with `std::string_view`?", "A string view does not own its characters. It becomes dangling when the referenced string or buffer is destroyed, reallocated, or otherwise invalidated. It is excellent for read-only parameters and slices when the owner outlives the view, but it must not silently escape that lifetime."),
            ("templates", "How do C++20 concepts improve template interfaces?", "Concepts state constraints in the interface and participate in overload selection. They make intent explicit and usually produce diagnostics near the caller rather than deep inside template instantiation. They constrain operations, not nominal inheritance, so they support generic programming over behavior."),
            ("exceptions", "What is the basic exception-safety guarantee?", "If an operation throws, invariants remain valid and no resources leak, though observable state may have changed. The strong guarantee additionally leaves state unchanged, and the no-throw guarantee promises completion without exceptions. RAII is the foundation for all three."),
            ("containers", "Why is `std::vector` usually the default sequence container?", "Its contiguous storage gives low overhead, cache-friendly traversal, random access, and compatibility with span-like views and C APIs. Growth is amortized constant time. Choose another container only for a demonstrated requirement such as stable node addresses or efficient operations at both ends."),
            ("iterators", "What causes iterator invalidation in C++?", "Operations that relocate, erase, or replace container elements may invalidate iterators, pointers, and references. The exact rules differ by container and operation: vector reallocation invalidates all element references, while many node-container insertions preserve them. Code must consult and encode the relevant contract."),
            ("const", "What is the difference between `const T*` and `T* const`?", "`const T*` is a pointer through which `T` cannot be modified, though the pointer may point elsewhere. `T* const` is a pointer value that cannot be reseated, though it may modify `T`. `const T* const` applies both restrictions."),
            ("undefined-behavior", "Why can undefined behavior break code far from the original mistake?", "The optimizer assumes undefined behavior never occurs and may transform surrounding code using that premise. An out-of-bounds access, invalid lifetime, signed overflow, or data race can therefore invalidate reasoning well beyond one instruction. Sanitizers, warnings, safer abstractions, and tests help detect it but do not redefine it."),
            ("virtual", "When does a C++ base class need a virtual destructor?", "If an object may be deleted through a base pointer, the base destructor must be virtual so the derived destructor runs. A polymorphic interface commonly declares `virtual ~Base() = default`. A class not intended for polymorphic deletion can avoid the cost and make that intent explicit."),
            ("lambdas", "What does `[&]` versus `[=]` capture in a C++ lambda?", "`[&]` implicitly captures odr-used automatic variables by reference; `[=]` captures them by value. Reference captures can dangle if the closure outlives the variables, while value captures snapshot state and may copy expensive objects. Explicit captures are clearer when lifetime matters."),
            ("concurrency", "Why does `volatile` not make C++ data access thread-safe?", "`volatile` concerns accesses with externally observable behavior, such as some device memory; it does not provide atomicity, ordering, or synchronization between threads. Use `std::atomic`, mutexes, condition variables, and a defined happens-before relationship for concurrency."),
            ("spans", "What problem does `std::span` solve?", "A span is a non-owning view over a contiguous sequence, carrying a pointer and extent together. It lets a function accept arrays, vectors, and subranges without copying or using an error-prone pointer-plus-length pair. The underlying storage must outlive the span."),
        ],
    }
    return [
        Example("code", "concept", topic, question, answer, language, "foundational")
        for language, items in rows.items()
        for topic, question, answer in items
    ]


ALGORITHMS = [
    (
        "frequency-table",
        "Write a function that counts words case-insensitively while preserving punctuation as part of each whitespace-delimited word.",
        {
            "python": "def word_counts(text: str) -> dict[str, int]:\n    counts: dict[str, int] = {}\n    for word in text.casefold().split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts",
            "rust": "use std::collections::HashMap;\n\nfn word_counts(text: &str) -> HashMap<String, usize> {\n    let mut counts = HashMap::new();\n    for word in text.split_whitespace() {\n        *counts.entry(word.to_lowercase()).or_insert(0) += 1;\n    }\n    counts\n}",
            "cpp": "#include <algorithm>\n#include <cctype>\n#include <sstream>\n#include <string>\n#include <unordered_map>\n\nstd::unordered_map<std::string, std::size_t> word_counts(std::string text) {\n    std::transform(text.begin(), text.end(), text.begin(),\n                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });\n    std::istringstream input(text);\n    std::unordered_map<std::string, std::size_t> counts;\n    for (std::string word; input >> word;) { ++counts[word]; }\n    return counts;\n}",
        },
        "A hash map makes each update expected constant time. The stated tokenization deliberately uses whitespace only; production natural-language tokenization needs a more precise policy.",
    ),
    (
        "balanced-delimiters",
        "Write a function that validates properly nested `()`, `[]`, and `{}` while ignoring other characters.",
        {
            "python": "def balanced(text: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack: list[str] = []\n    for char in text:\n        if char in pairs.values():\n            stack.append(char)\n        elif char in pairs:\n            if not stack or stack.pop() != pairs[char]:\n                return False\n    return not stack",
            "rust": "fn balanced(text: &str) -> bool {\n    let mut stack = Vec::new();\n    for ch in text.chars() {\n        match ch {\n            '(' | '[' | '{' => stack.push(ch),\n            ')' | ']' | '}' => {\n                let expected = match ch { ')' => '(', ']' => '[', _ => '{' };\n                if stack.pop() != Some(expected) { return false; }\n            }\n            _ => {}\n        }\n    }\n    stack.is_empty()\n}",
            "cpp": "#include <string_view>\n#include <vector>\n\nbool balanced(std::string_view text) {\n    std::vector<char> stack;\n    for (char ch : text) {\n        if (ch == '(' || ch == '[' || ch == '{') stack.push_back(ch);\n        else if (ch == ')' || ch == ']' || ch == '}') {\n            const char expected = ch == ')' ? '(' : ch == ']' ? '[' : '{';\n            if (stack.empty() || stack.back() != expected) return false;\n            stack.pop_back();\n        }\n    }\n    return stack.empty();\n}",
        },
        "A stack records unmatched opening delimiters. A closer must match the most recent opener, and a valid input leaves the stack empty.",
    ),
    (
        "binary-search",
        "Implement lower bound: return the first index whose value is not less than the target in an ascending integer sequence.",
        {
            "python": "def lower_bound(values: list[int], target: int) -> int:\n    low, high = 0, len(values)\n    while low < high:\n        middle = low + (high - low) // 2\n        if values[middle] < target:\n            low = middle + 1\n        else:\n            high = middle\n    return low",
            "rust": "fn lower_bound(values: &[i32], target: i32) -> usize {\n    let (mut low, mut high) = (0, values.len());\n    while low < high {\n        let middle = low + (high - low) / 2;\n        if values[middle] < target { low = middle + 1; }\n        else { high = middle; }\n    }\n    low\n}",
            "cpp": "#include <cstddef>\n#include <span>\n\nstd::size_t lower_bound(std::span<const int> values, int target) {\n    std::size_t low = 0, high = values.size();\n    while (low < high) {\n        const auto middle = low + (high - low) / 2;\n        if (values[middle] < target) low = middle + 1;\n        else high = middle;\n    }\n    return low;\n}",
        },
        "The invariant is that every index below `low` is too small and no candidate at or above `high` has been discarded. A half-open interval also handles empty input naturally.",
    ),
    (
        "run-length-encoding",
        "Implement run-length encoding that turns consecutive equal characters into `(character, count)` pairs. Empty input should produce an empty result.",
        {
            "python": "def run_lengths(text: str) -> list[tuple[str, int]]:\n    result: list[tuple[str, int]] = []\n    for char in text:\n        if result and result[-1][0] == char:\n            previous, count = result[-1]\n            result[-1] = (previous, count + 1)\n        else:\n            result.append((char, 1))\n    return result",
            "rust": "fn run_lengths(text: &str) -> Vec<(char, usize)> {\n    let mut result: Vec<(char, usize)> = Vec::new();\n    for ch in text.chars() {\n        match result.last_mut() {\n            Some((last, count)) if *last == ch => *count += 1,\n            _ => result.push((ch, 1)),\n        }\n    }\n    result\n}",
            "cpp": "#include <string_view>\n#include <utility>\n#include <vector>\n\nstd::vector<std::pair<char, std::size_t>> run_lengths(std::string_view text) {\n    std::vector<std::pair<char, std::size_t>> result;\n    for (char ch : text) {\n        if (!result.empty() && result.back().first == ch) ++result.back().second;\n        else result.emplace_back(ch, 1);\n    }\n    return result;\n}",
        },
        "The algorithm makes one pass and stores one entry per run, so its time is linear and auxiliary space is proportional to the number of runs.",
    ),
    (
        "sliding-window",
        "Return the largest sum of any contiguous window of exactly `width` integers. Reject width zero or a width larger than the input.",
        {
            "python": "def max_window_sum(values: list[int], width: int) -> int:\n    if width <= 0 or width > len(values):\n        raise ValueError('invalid window width')\n    current = sum(values[:width])\n    best = current\n    for right in range(width, len(values)):\n        current += values[right] - values[right - width]\n        best = max(best, current)\n    return best",
            "rust": "fn max_window_sum(values: &[i64], width: usize) -> Option<i64> {\n    if width == 0 || width > values.len() { return None; }\n    let mut current: i64 = values[..width].iter().sum();\n    let mut best = current;\n    for right in width..values.len() {\n        current += values[right] - values[right - width];\n        best = best.max(current);\n    }\n    Some(best)\n}",
            "cpp": "#include <algorithm>\n#include <cstddef>\n#include <optional>\n#include <span>\n\nstd::optional<long long> max_window_sum(std::span<const int> values, std::size_t width) {\n    if (width == 0 || width > values.size()) return std::nullopt;\n    long long current = 0;\n    for (std::size_t i = 0; i < width; ++i) current += values[i];\n    long long best = current;\n    for (std::size_t right = width; right < values.size(); ++right) {\n        current += values[right] - values[right - width];\n        best = std::max(best, current);\n    }\n    return best;\n}",
        },
        "After computing the first window, each step adds the entering value and subtracts the leaving value. That reduces the time from O(n × width) to O(n).",
    ),
    (
        "deduplicate-stable",
        "Remove duplicate integers while preserving their first-seen order.",
        {
            "python": "def stable_unique(values: list[int]) -> list[int]:\n    seen: set[int] = set()\n    result = []\n    for value in values:\n        if value not in seen:\n            seen.add(value)\n            result.append(value)\n    return result",
            "rust": "use std::collections::HashSet;\n\nfn stable_unique(values: &[i32]) -> Vec<i32> {\n    let mut seen = HashSet::new();\n    values.iter().copied().filter(|value| seen.insert(*value)).collect()\n}",
            "cpp": "#include <span>\n#include <unordered_set>\n#include <vector>\n\nstd::vector<int> stable_unique(std::span<const int> values) {\n    std::unordered_set<int> seen;\n    std::vector<int> result;\n    result.reserve(values.size());\n    for (int value : values) if (seen.insert(value).second) result.push_back(value);\n    return result;\n}",
        },
        "The set answers whether a value has appeared, while the separate result retains encounter order. Expected time is linear.",
    ),
    (
        "merge-intervals",
        "Merge overlapping closed integer intervals. The input may be unsorted; reject intervals whose start exceeds their end.",
        {
            "python": "def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n    if any(start > end for start, end in intervals):\n        raise ValueError('reversed interval')\n    result: list[tuple[int, int]] = []\n    for start, end in sorted(intervals):\n        if not result or start > result[-1][1]:\n            result.append((start, end))\n        else:\n            old_start, old_end = result[-1]\n            result[-1] = (old_start, max(old_end, end))\n    return result",
            "rust": "fn merge_intervals(mut intervals: Vec<(i32, i32)>) -> Option<Vec<(i32, i32)>> {\n    if intervals.iter().any(|(start, end)| start > end) { return None; }\n    intervals.sort_unstable();\n    let mut result: Vec<(i32, i32)> = Vec::new();\n    for (start, end) in intervals {\n        match result.last_mut() {\n            Some((_, previous_end)) if start <= *previous_end => *previous_end = (*previous_end).max(end),\n            _ => result.push((start, end)),\n        }\n    }\n    Some(result)\n}",
            "cpp": "#include <algorithm>\n#include <optional>\n#include <utility>\n#include <vector>\n\nusing Interval = std::pair<int, int>;\nstd::optional<std::vector<Interval>> merge_intervals(std::vector<Interval> intervals) {\n    for (auto [start, end] : intervals) if (start > end) return std::nullopt;\n    std::sort(intervals.begin(), intervals.end());\n    std::vector<Interval> result;\n    for (auto [start, end] : intervals) {\n        if (result.empty() || start > result.back().second) result.emplace_back(start, end);\n        else result.back().second = std::max(result.back().second, end);\n    }\n    return result;\n}",
        },
        "Sorting by start makes every possible overlap occur next to the interval currently being built. Sorting dominates the O(n log n) running time.",
    ),
    (
        "breadth-first-search",
        "Given an adjacency list and a start node, return shortest unweighted distances to every reachable node.",
        {
            "python": "from collections import deque\n\ndef distances(graph: dict[str, list[str]], start: str) -> dict[str, int]:\n    result = {start: 0}\n    queue = deque([start])\n    while queue:\n        node = queue.popleft()\n        for neighbor in graph.get(node, []):\n            if neighbor not in result:\n                result[neighbor] = result[node] + 1\n                queue.append(neighbor)\n    return result",
            "rust": "use std::collections::{HashMap, VecDeque};\n\nfn distances(graph: &HashMap<String, Vec<String>>, start: &str) -> HashMap<String, usize> {\n    let mut result = HashMap::from([(start.to_owned(), 0)]);\n    let mut queue = VecDeque::from([start.to_owned()]);\n    while let Some(node) = queue.pop_front() {\n        let next_distance = result[&node] + 1;\n        for neighbor in graph.get(&node).into_iter().flatten() {\n            if !result.contains_key(neighbor) {\n                result.insert(neighbor.clone(), next_distance);\n                queue.push_back(neighbor.clone());\n            }\n        }\n    }\n    result\n}",
            "cpp": "#include <queue>\n#include <string>\n#include <unordered_map>\n#include <vector>\n\nusing Graph = std::unordered_map<std::string, std::vector<std::string>>;\nstd::unordered_map<std::string, int> distances(const Graph& graph, const std::string& start) {\n    std::unordered_map<std::string, int> result{{start, 0}};\n    std::queue<std::string> queue; queue.push(start);\n    while (!queue.empty()) {\n        auto node = queue.front(); queue.pop();\n        auto found = graph.find(node);\n        if (found == graph.end()) continue;\n        for (const auto& neighbor : found->second) {\n            if (!result.contains(neighbor)) {\n                result[neighbor] = result[node] + 1;\n                queue.push(neighbor);\n            }\n        }\n    }\n    return result;\n}",
        },
        "A FIFO queue visits nodes in nondecreasing path length. Recording a node when it is enqueued prevents repeated work and makes the first recorded distance shortest.",
    ),
    (
        "parse-key-values",
        "Parse lines of `key=value`, ignoring blank lines and lines whose first non-space character is `#`. Trim key and value; reject missing or empty keys.",
        {
            "python": "def parse_pairs(text: str) -> dict[str, str]:\n    result = {}\n    for number, raw in enumerate(text.splitlines(), 1):\n        line = raw.strip()\n        if not line or line.startswith('#'):\n            continue\n        if '=' not in line:\n            raise ValueError(f'line {number}: missing =')\n        key, value = (part.strip() for part in line.split('=', 1))\n        if not key:\n            raise ValueError(f'line {number}: empty key')\n        result[key] = value\n    return result",
            "rust": "use std::collections::HashMap;\n\nfn parse_pairs(text: &str) -> Result<HashMap<String, String>, String> {\n    let mut result = HashMap::new();\n    for (index, raw) in text.lines().enumerate() {\n        let line = raw.trim();\n        if line.is_empty() || line.starts_with('#') { continue; }\n        let (key, value) = line.split_once('=').ok_or_else(|| format!(\"line {}: missing =\", index + 1))?;\n        if key.trim().is_empty() { return Err(format!(\"line {}: empty key\", index + 1)); }\n        result.insert(key.trim().to_owned(), value.trim().to_owned());\n    }\n    Ok(result)\n}",
            "cpp": "#include <algorithm>\n#include <cctype>\n#include <sstream>\n#include <stdexcept>\n#include <string>\n#include <unordered_map>\n\nstd::string trim(std::string value) {\n    auto nonspace = [](unsigned char c) { return !std::isspace(c); };\n    value.erase(value.begin(), std::find_if(value.begin(), value.end(), nonspace));\n    value.erase(std::find_if(value.rbegin(), value.rend(), nonspace).base(), value.end());\n    return value;\n}\n\nstd::unordered_map<std::string, std::string> parse_pairs(const std::string& text) {\n    std::unordered_map<std::string, std::string> result;\n    std::istringstream input(text); std::string line; int number = 0;\n    while (std::getline(input, line)) {\n        ++number; line = trim(line);\n        if (line.empty() || line.front() == '#') continue;\n        auto equal = line.find('=');\n        if (equal == std::string::npos) throw std::invalid_argument(\"missing = on line \" + std::to_string(number));\n        auto key = trim(line.substr(0, equal));\n        if (key.empty()) throw std::invalid_argument(\"empty key on line \" + std::to_string(number));\n        result[key] = trim(line.substr(equal + 1));\n    }\n    return result;\n}",
        },
        "Splitting only at the first equals sign allows values to contain additional equals signs. Line-numbered errors make malformed configuration easy to locate.",
    ),
]


def algorithm_examples() -> list[Example]:
    names = {"python": "Python", "rust": "Rust", "cpp": "C++20"}
    tags = {"python": "python", "rust": "rust", "cpp": "cpp"}
    examples = []
    for topic, prompt, implementations, explanation in ALGORITHMS:
        for language, code in implementations.items():
            answer = explanation + "\n\n" + fenced(tags[language], code)
            examples.append(
                Example("code", "implementation", topic, f"In {names[language]}, {prompt}", answer, language)
            )
    return examples


def debugging_examples() -> list[Example]:
    rows = [
        ("python", "mutation-during-iteration", "Why does this skip adjacent matches, and how should it be fixed?\n\n" + fenced("python", "values = [2, 4, 5, 6]\nfor value in values:\n    if value % 2 == 0:\n        values.remove(value)"), "Removing shifts later elements while the iterator advances. Build a new list when replacement is intended:\n\n" + fenced("python", "values = [value for value in values if value % 2 != 0]")),
        ("python", "late-bound-closure", "Fix these callbacks so they return 0, 1, and 2 rather than all returning 2.\n\n" + fenced("python", "callbacks = [lambda: index for index in range(3)]"), "Closures look up `index` when called. Bind the current value as a default argument during construction:\n\n" + fenced("python", "callbacks = [lambda index=index: index for index in range(3)]")),
        ("python", "exception-scope", "Why is `except Exception: return None` often a bad parser implementation?", "It collapses programming bugs, I/O failures, and expected bad input into the same result. Catch the narrow documented exception, such as `ValueError`, and let unrelated failures retain their traceback. If absence is valid, distinguish it explicitly from malformed input."),
        ("python", "shallow-copy", "A nested list changes through both the original and its copy. Explain and fix it.", "A list copy duplicates only the outer container, so nested mutable objects remain shared. Construct the required nested values independently or use `copy.deepcopy` when arbitrary object graphs truly need independent copies. Often a purpose-built copy method makes ownership clearer."),
        ("rust", "borrow-loop", "Why can this pattern fail to compile: borrowing an element from a vector, then pushing to the vector before the borrow's last use?", "`push` may reallocate the vector, which would invalidate a reference into its buffer. End the borrow before mutation, store an index or owned clone when appropriate, or restructure the operation into separate phases. The compiler is preventing a potential dangling reference."),
        ("rust", "mutex-guard", "A Rust thread deadlocks when it locks the same non-reentrant `Mutex` twice. What is the usual fix?", "Limit the first `MutexGuard`'s scope or explicitly drop it before taking the second lock. Better, perform all related work under one acquisition or move the inner operation to a helper that accepts `&mut T` rather than locking again. Keep lock ordering consistent across multiple mutexes."),
        ("rust", "temporary-reference", "Why can't a Rust function return `&String` pointing to a `String` created inside that function?", "The local `String` is dropped when the function returns, so the reference would dangle. Return the owned `String`, borrow from an input whose lifetime can be related to the output, or use an owned/shared container appropriate to the API."),
        ("rust", "clone-overuse", "A Rust program adds `.clone()` until borrow errors disappear. Why is that a design smell?", "Cloning can be correct, but reflexive cloning hides ownership decisions and may add allocations. First decide who should own the value, which callers only need a borrow, and whether work can be reordered to shorten borrows. Clone where independent ownership is part of the contract."),
        ("cpp", "dangling-view", "Find the lifetime bug: a function constructs a local `std::string` and returns `std::string_view` to it.", "The view points into storage destroyed at return, so every use is undefined behavior. Return `std::string` by value, or return a view only into storage owned by the caller or another object whose lifetime is explicitly longer."),
        ("cpp", "erase-iterator", "How should a C++ loop erase selected elements from a vector without using an invalidated iterator?", "Use the iterator returned by `erase` and advance only when retaining an element, or use `std::erase_if` in C++20. `erase` invalidates iterators at and after the erased position, so blindly incrementing the old iterator is invalid."),
        ("cpp", "shared-cycle", "Why can two objects holding `shared_ptr` to each other leak?", "Each strong reference keeps the other's count above zero, so neither destructor runs after outside owners disappear. Model one back-edge or observer relationship with `weak_ptr`, then lock it temporarily when access is needed."),
        ("cpp", "uninitialized", "Why is reading an uninitialized local integer in C++ not merely an unpredictable value?", "For ordinary automatic integers, the value is indeterminate and reading it can be undefined behavior. Initialize variables at declaration and let warnings or sanitizers catch missed paths. Do not use an apparent debug-build value as evidence of defined behavior."),
    ]
    return [Example("code", "debugging", topic, question, answer, language) for language, topic, question, answer in rows]


def comparison_examples() -> list[Example]:
    rows = [
        ("ownership", "Compare how Python, Rust, and C++ express ownership of a heap-backed sequence.", "Python variables normally hold references to garbage-collected objects, and aliasing is common. Rust's `Vec<T>` has one owner by default; moves transfer ownership and borrows provide checked temporary access. C++ `std::vector<T>` is also an owning value, but references and pointers rely more heavily on programmer-observed lifetime and invalidation rules. All three favor returning the owning container by value."),
        ("errors", "Compare routine error handling in Python, Rust, and modern C++.", "Python uses exceptions and requires conventions about which failures an API raises. Rust makes recoverable failure explicit with `Result<T, E>` and uses `?` for propagation; panics are generally for violated invariants or unrecoverable states. C++ supports exceptions but codebases vary, sometimes using expected-like result types. In every language, preserve context and avoid silently swallowing unrelated errors."),
        ("polymorphism", "Compare interfaces in Python, Rust, and C++ without focusing on inheritance.", "Python commonly uses duck typing, optionally documented with `Protocol`. Rust uses traits with static dispatch through generics or dynamic dispatch through trait objects. C++ uses templates/concepts for static polymorphism and virtual functions for runtime polymorphism. The shared design lesson is to depend on the smallest behavior the consumer needs."),
        ("resource-cleanup", "How do Python, Rust, and C++ make resource cleanup reliable?", "Python uses context managers for deterministic scoped cleanup, while garbage collection alone is not a good timing contract. Rust drops owned values automatically at scope exit and uses guard types. C++ uses RAII destructors. Each approach works best when resource ownership is represented by an object rather than scattered acquire/release calls."),
        ("nullability", "Compare `None`, `Option`, `std::optional`, and null pointers.", "Python's `None` is a normal runtime value and annotations can document optionality. Rust's `Option<T>` forces callers to handle presence and absence through the type system. C++ `std::optional<T>` does the same for values, while a null pointer may also express non-ownership or optional identity. Prefer the representation that states the API contract without sentinel ambiguity."),
        ("iteration", "What common idea connects Python iterators, Rust iterators, and C++ ranges?", "They separate traversal from the underlying container and allow algorithms to compose around sequences. Python emphasizes a runtime iterator protocol, Rust uses lazy strongly typed adapters that optimize through monomorphization, and C++ ranges add views and constrained algorithms. In all three, avoid materializing intermediates unless ownership or repeated access requires it."),
        ("immutability", "How does each language communicate that a function should not mutate an input?", "Python relies on conventions, immutable types, protocols, and tests; annotations do not generally enforce mutation rules. Rust uses shared references `&T` versus exclusive `&mut T`. C++ uses `const T&`, although logical constness and interior mutation remain possible. Clear APIs should also specify aliasing and lifetime expectations."),
        ("concurrency", "Compare the most important shared-memory concurrency hazard across Python, Rust, and C++.", "All three can have logical races and deadlocks. CPython's GIL does not make compound operations or external resources automatically safe. Rust prevents many data races through `Send`, `Sync`, and borrowing but cannot prevent deadlock. C++ data races are undefined behavior and require disciplined synchronization. Minimize shared mutable state in every language."),
    ]
    return [Example("code", "cross_language", topic, question, answer, "multiple") for topic, question, answer in rows]


def linux_examples() -> list[Example]:
    rows = [
        ("filesystem", "How can I find which files under the current directory use the most disk space without modifying anything?", "Use an observation-only pipeline such as:\n\n" + fenced("bash", "find . -type f -printf '%s\\t%p\\n' | sort -n | tail -20") + "\n\n`find` prints byte size and path, numeric `sort` orders them, and `tail` selects the largest entries. Paths containing newlines make line-oriented output ambiguous; for adversarial filenames use a NUL-delimited toolchain or a small script."),
        ("filesystem", "What is the difference between a hard link and a symbolic link?", "A hard link is another directory entry for the same inode; it normally cannot cross filesystems and the data remains until the last link is removed. A symbolic link stores a path and may cross filesystems or become dangling. Permissions on access are determined by the target, while replacing the target path may make a symlink resolve to a different inode."),
        ("filesystem", "How do I inspect a path component by component when permissions cause `Permission denied`?", "Run `namei -l /full/path/to/item`. It displays every component with type, owner, group, and mode, which reveals the directory missing execute/search permission. `ls -l` on only the final path can hide a failure higher in the tree."),
        ("filesystem", "Why can disk space remain used after a large log file is deleted?", "A running process may still have the unlinked inode open. The directory entry is gone, but storage is released only after the final file descriptor closes. Inspect candidates with `lsof +L1` or `/proc/*/fd`; then arrange for the owning service to reopen or close the file rather than deleting process state blindly."),
        ("filesystem", "What is a safe pattern for making a temporary directory in a shell script?", "Use the system allocator and ensure cleanup is quoted:\n\n" + fenced("bash", "tmp_dir=$(mktemp -d)\ntrap 'rm -rf -- \"$tmp_dir\"' EXIT HUP INT TERM") + "\n\nDo not invent a predictable `/tmp/name.$$` path. Keep the variable immutable, validate it if later logic can change it, and avoid running the cleanup as elevated privilege unless necessary."),
        ("permissions", "Explain read, write, and execute bits on a directory.", "Directory read permits listing names, write permits creating/removing/renaming entries, and execute permits searching or traversing to named entries. Useful operations often need combinations: listing metadata usually needs read plus execute, while deleting a file depends mainly on write and execute permissions on its parent directory, not write permission on the file itself."),
        ("permissions", "What do setuid, setgid, and the sticky bit mean on Linux files and directories?", "Setuid on an executable can run it with the file owner's effective UID; setgid similarly affects the effective GID. Setgid on a directory makes new entries inherit the directory's group. The sticky bit on a shared directory restricts removal/renaming to appropriate owners or privileged processes. Scripts generally do not receive setuid semantics on Linux."),
        ("permissions", "Why is `chmod -R 777` a poor response to a permissions problem?", "It grants every local user broad write access, may expose secrets, and removes the distinction between executable and non-executable files. Diagnose the required owner, group, directory traversal, and service identity. Then apply the narrow mode—often a group, ACL, or corrected ownership—only to the intended paths."),
        ("permissions", "When are POSIX ACLs useful compared with traditional owner/group/other mode bits?", "ACLs grant specific users or groups access without reorganizing the primary ownership model. Inspect them with `getfacl` and change them with `setfacl`. Remember the ACL mask limits effective permissions for named users/groups, and default ACLs on directories influence newly created children."),
        ("processes", "How can I identify what is listening on TCP port 8080?", "Use `ss -ltnp 'sport = :8080'`. `-l` selects listeners, `-t` TCP, `-n` numeric addresses, and `-p` process details. Process information for other users may require privilege. Confirm the network namespace if containers are involved."),
        ("processes", "What is the difference between SIGTERM and SIGKILL?", "SIGTERM requests termination and can be caught so a process flushes state and releases resources. SIGKILL cannot be caught, blocked, or handled; the kernel stops the process immediately. Send TERM first, wait an appropriate timeout, investigate if it remains, and reserve KILL for a process that cannot shut down normally."),
        ("processes", "Why can a process be shown as a zombie, and how is it removed?", "A zombie has exited but its parent has not collected the exit status with a wait operation. It consumes a process-table entry but no running memory image. The parent must reap it; killing the zombie cannot help because it is already dead. Fix or restart the parent, or let reparenting move it to a proper reaper."),
        ("processes", "How can I observe which files a process currently has open?", "Inspect `/proc/PID/fd/` for descriptor symlinks, or use `lsof -p PID` for a richer decoded view. Access is subject to ownership, ptrace restrictions, namespaces, and privilege. Treat the result as a snapshot because descriptors can change immediately."),
        ("processes", "What do Linux load averages measure?", "They approximate the average number of tasks runnable on CPUs or waiting in uninterruptible sleep, traditionally including many I/O waits, over 1, 5, and 15 minutes. They are not percentages. Interpret them relative to CPU count and corroborate with CPU, I/O, memory-pressure, and per-process metrics."),
        ("systemd", "How should I investigate a systemd service that failed to start?", "Start with `systemctl status --no-pager SERVICE` for state and recent messages, then `journalctl -u SERVICE -b --no-pager` for this boot's unit log. Use `systemctl cat SERVICE` to see the effective unit and drop-ins, and `systemd-analyze verify` for unit-file diagnostics. Read before restarting so evidence is not obscured."),
        ("systemd", "What is the difference between `systemctl reload`, `restart`, and `daemon-reload`?", "`reload` asks a running service to reread its own configuration if supported. `restart` stops and starts the service, interrupting it. `daemon-reload` makes systemd reread unit definitions and generators; it does not by itself reload the application's configuration or restart the service."),
        ("systemd", "How can a systemd service receive a secret without putting it directly in the unit file?", "Prefer the platform's credential mechanism (`LoadCredential=`/encrypted credentials) when available, or a tightly permissioned file referenced by the service. Environment variables and command lines can leak through introspection or logs. Define ownership, rotation, restart behavior, and which service sandbox paths can access the secret."),
        ("logs", "How do I view kernel messages from the current boot with human-readable timestamps?", "Use `journalctl -k -b -o short-iso` on a journal-based system. `-k` selects kernel messages and `-b` the current boot. `dmesg --time-format iso` is another view but access may be restricted and the ring buffer can wrap."),
        ("logs", "Why is `tail -f` sometimes insufficient for following a rotated log?", "Following by descriptor stays attached to the old inode when a rotator renames the file and the application opens a new one. GNU `tail -F` follows by name and retries, which handles common rotations. For managed services, querying the journal or logging backend is often more reliable."),
        ("networking", "How can I distinguish a DNS failure from a TCP connection failure?", "Resolve first with a tool such as `getent ahosts NAME` or `dig NAME`, noting which resolver path the application uses. Then test the resolved endpoint and port with `nc -vz HOST PORT` or an application-aware client. `ip route get ADDRESS` shows routing. Avoid concluding that ping failure means the service is down because ICMP may be filtered."),
        ("networking", "What information does `ip route get 203.0.113.10` provide?", "It asks the kernel which route would be selected for that destination, including output interface, gateway, preferred source address, and routing-table decisions. It does not send a packet or prove the remote endpoint is reachable, but it is more precise than visually scanning the route table."),
        ("networking", "Why might `localhost` work while the machine's LAN address does not?", "The service may be bound only to loopback, a firewall may reject the external path, the client may be in another network namespace, or address-family selection may differ. Inspect listeners with `ss -ltnp`, the service bind configuration, namespace/container port publishing, and firewall rules."),
        ("networking", "What is the difference between a network interface being up and having working connectivity?", "The administrative and carrier states only describe local interface readiness. Connectivity additionally requires correct addresses, routes, neighbor discovery, firewall policy, DNS when names are used, and a reachable remote service. Diagnose layer by layer rather than treating `UP` as an end-to-end test."),
        ("storage", "How do `df` and `du` answer different disk-usage questions?", "`df` reports filesystem allocation from filesystem metadata. `du` walks visible directory entries and totals their blocks. They can disagree because of deleted-but-open files, mount points hiding data, sparse files, snapshots, reflinks, permissions, or reserved blocks. Use both and investigate the reason rather than assuming one is wrong."),
        ("storage", "What should I check before growing a Linux filesystem?", "Identify the full stack: filesystem type and current state, mount, logical volume or partition, encryption/device mapper, and underlying device capacity. Back up important data, consult the filesystem's supported online/offline procedure, grow lower layers before upper layers, and never guess device names from a copied command."),
        ("storage", "What is inode exhaustion and how does it differ from running out of bytes?", "Filesystems allocate both data blocks and inodes. Millions of tiny files can consume all inodes while byte capacity remains. Check `df -i`; locate high file-count directories carefully. Recovery requires removing or consolidating entries, and some filesystems require recreation to change inode provisioning."),
        ("shell", "Why should shell variable expansions usually be double-quoted?", "Without quotes, parameter expansion is followed by word splitting and pathname expansion, so whitespace and wildcard characters change the argument list. Use `\"$value\"` and `\"${array[@]}\"`. Deliberate splitting should be explicit and documented rather than an accidental consequence of unquoted data."),
        ("shell", "What does `set -euo pipefail` help with, and why is it not complete error handling?", "`-u` catches many unset variables, `pipefail` exposes failures before the final pipeline command, and `-e` exits in several unhandled-error contexts. But `-e` has conditional and version-sensitive semantics, expected nonzero statuses still need handling, and cleanup requires traps. Explicit checks remain important at decision points."),
        ("shell", "How can a shell script safely process arbitrary filenames produced by `find`?", "Use NUL delimiters end to end, for example:\n\n" + fenced("bash", "find root -type f -print0 | while IFS= read -r -d '' path; do\n    printf '%q\\n' \"$path\"\ndone") + "\n\nNewline-delimited loops and `for path in $(find ...)` corrupt names containing whitespace, glob characters, or newlines."),
        ("shell", "Why is parsing `ls` output in a script unreliable?", "Its display is for humans: quoting, columns, colors, locale ordering, and control-character handling can change, and filenames may contain newlines. Use globs, `find -print0`, shell arrays, or APIs that preserve argument boundaries."),
        ("security", "What is the principle of least privilege on Linux?", "Give a process only the identities, capabilities, filesystem paths, devices, network access, and duration of privilege needed for its task. Prefer a dedicated service account and narrow systemd/container sandboxing over unrestricted root. Verify the effective boundary rather than trusting one configuration layer."),
        ("capabilities", "Why were Linux capabilities introduced?", "Traditional Unix treated UID 0 as one all-powerful privilege. Linux capabilities split many root checks into independently assignable units such as binding low ports or changing network configuration. This can reduce authority, but capabilities are coarse and some—especially `CAP_SYS_ADMIN`—remain extremely powerful."),
        ("capabilities", "What are the permitted, effective, and inheritable capability sets?", "Permitted is the ceiling of capabilities a thread may make effective. Effective capabilities are currently used for permission checks. Inheritable capabilities participate in capability calculation across `execve` for suitably marked executables. Linux also has bounding and ambient sets, which further limit or preserve capabilities across execution."),
        ("capabilities", "What does the capability bounding set do?", "It limits which capabilities a process and its descendants can gain across executable-file transitions. Dropping a capability from the bounding set is a one-way reduction for that process tree without higher-level recreation. It is a useful defense-in-depth ceiling, not a grant of capability by itself."),
        ("capabilities", "What are ambient capabilities for?", "Ambient capabilities let a non-root process preserve selected capabilities across `execve` of ordinary non-privileged programs. A capability must already be permitted and inheritable before it can be raised ambiently. Executing a setuid/setgid or file-capability program clears the ambient set to prevent surprising privilege combinations."),
        ("capabilities", "Why should `CAP_SYS_ADMIN` be avoided when a narrower capability exists?", "It gates a very broad and historically growing collection of privileged operations, making it close to a catch-all. Granting it can undermine namespace and container isolation assumptions. Determine the exact failing operation and use a narrower capability or redesign the operation when possible."),
        ("capabilities", "How can I inspect a process's Linux capability sets?", "Use `getpcaps PID` when libcap tools are installed, and inspect `CapInh`, `CapPrm`, `CapEff`, `CapBnd`, and `CapAmb` in `/proc/PID/status`. `capsh --decode=HEX` can decode masks. Account for user namespaces, which scope capability meaning to the namespace's governed resources."),
        ("capabilities", "How do file capabilities differ from making an executable setuid root?", "File capabilities attach selected privilege bits to an executable, potentially granting only what it needs after `execve`. Setuid root changes effective identity and often grants a much broader path to privilege. File capabilities still require careful threat modeling: every code path, loaded input, and invoked helper runs inside the resulting authority context."),
        ("capabilities", "What does `CAP_NET_BIND_SERVICE` allow?", "It permits binding Internet-domain sockets to privileged ports below 1024 in the governing user namespace. It does not grant arbitrary firewall administration, raw packet access, or general network configuration. An alternative is to have a privileged front end proxy to an unprivileged service."),
        ("capabilities", "What is `CAP_NET_ADMIN`, and why is it risky?", "It authorizes many network-administration operations such as interface configuration, routing and firewall changes, traffic control, and some socket options. That is a wide attack surface. Scope it with network/user namespaces when appropriate and do not grant it merely because an application needs ordinary network access."),
        ("capabilities", "What does `no_new_privs` protect against?", "Once set for a thread, `no_new_privs` prevents `execve` from granting privileges the caller did not already have through setuid/setgid bits or file capabilities. It is inherited and cannot be unset. Sandboxes use it before installing unprivileged seccomp filters, but it does not remove privileges already held."),
        ("capabilities", "Why might a file capability appear present but not take effect?", "Possible causes include `no_new_privs`, a bounding-set restriction, execution from a filesystem mounted `nosuid`, namespace rules, an interpreter/script transition, or a capability-unaware execution path. Inspect the file with `getcap`, the process sets in `/proc`, mount options, and the exact `execve` chain."),
        ("containers", "Why is root inside a user namespace not necessarily host root?", "A user namespace maps namespace UIDs/GIDs to different host IDs and scopes capabilities to resources governed by that namespace. UID 0 inside may have broad authority there but little outside. Bind mounts, devices, kernel interfaces, and incorrectly granted capabilities can still expose host resources, so the boundary must be evaluated as a whole."),
        ("containers", "What is the difference between a namespace and a cgroup?", "Namespaces change what a process can see and identify—such as PIDs, mounts, users, networks, IPC, or hostnames. Cgroups organize processes for resource accounting, limits, prioritization, and control. Containers commonly combine both, plus capabilities, seccomp, LSM policy, and filesystem isolation."),
        ("containers", "Why is mounting the container runtime socket into a container dangerous?", "The socket often controls creation of new containers with chosen mounts, devices, and privileges. A process that can use it may ask the runtime to expose the host filesystem or launch a highly privileged workload, effectively gaining host-level control. Treat socket access as administrative authority."),
        ("diagnostics", "A command works interactively but fails in cron or a service. What should I compare?", "Compare user/group identity, working directory, environment and `PATH`, shell, umask, resource limits, namespaces, mandatory access control, available mounts, terminal assumptions, and secret/config injection. Use absolute paths and log the effective context; do not solve it by copying the entire interactive environment blindly."),
        ("diagnostics", "How can `strace` help diagnose a failing program?", "It records system calls and results, exposing missing files, permission errors, connection attempts, signals, and blocking operations. Start narrowly, for example `strace -f -e trace=file,network -o trace.log COMMAND`. Traces can contain secrets and impose overhead, and attaching may require ptrace permission."),
        ("diagnostics", "What does an exit status of 126 versus 127 usually mean in a shell?", "127 conventionally means the command could not be found. 126 means it was found but could not be executed, perhaps due to permissions, format, mount options, or an invalid interpreter. Values above 128 often encode termination by a signal, but applications may define their own statuses."),
        ("packages", "Why is manually replacing a package-managed file risky?", "The package database no longer describes the installed state, upgrades may overwrite the change, verification reports drift, dependencies may become inconsistent, and rollback is unclear. Prefer supported configuration, a local package, alternatives/overrides, or a separately installed path with explicit ownership."),
        ("backups", "What makes a backup trustworthy?", "A backup must have a defined scope, consistent capture method, retention, access protection, and an independently tested restore procedure. Check application consistency and metadata—not just copied bytes. Monitor failures, keep at least one failure-domain-separated copy, and periodically restore representative data into a clean environment."),
    ]
    return [Example("linux", "instruction", topic, question, answer, None) for topic, question, answer in rows]


def all_examples() -> list[Example]:
    return concepts() + algorithm_examples() + debugging_examples() + comparison_examples() + linux_examples()


def validate(examples: list[Example]) -> None:
    keys = [(item.dataset, item.category, item.language, item.topic, item.question) for item in examples]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate example identity")
    pairs = [(item.question, item.answer) for item in examples]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate question/answer pair")
    for item in examples:
        if not item.question.strip() or not item.answer.strip():
            raise ValueError(f"empty content in {item.topic}")
        if item.dataset == "code" and item.language not in {"python", "rust", "cpp", "multiple"}:
            raise ValueError(f"bad language in {item.topic}")
        for code in re.findall(r"```python\n(.*?)```", item.question + item.answer, re.DOTALL):
            try:
                ast.parse(code)
            except SyntaxError as error:
                raise ValueError(f"invalid Python in {item.topic}: {error}") from error


def record(example: Example, ordinal: int) -> dict:
    identity = f"{example.dataset}_{example.category}_{ordinal:04d}"
    metadata = {
        "category": example.category,
        "topic": example.topic,
        "difficulty": example.difficulty,
        "license": "CC0-1.0",
        "origin": "synthetic: authored for llm_bobgpt",
    }
    if example.language:
        metadata["language"] = example.language
    return {
        "id": f"{example.dataset}_tutor/{identity}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[example.dataset]},
            {"role": "user", "content": example.question},
            {"role": "assistant", "content": example.answer},
        ],
        "metadata": metadata,
    }


def write_dataset(root: Path, dataset: str, examples: list[Example]) -> None:
    output_dir = root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [record(item, index) for index, item in enumerate(examples, 1)]
    (output_dir / f"{dataset}.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8"
    )
    categories = Counter(item.category for item in examples)
    languages = Counter(item.language for item in examples if item.language)
    metadata = {
        "description": "Distinct authored instructional conversations; no mix-level repetition.",
        "license": "CC0-1.0",
        "generator": "scripts/generate_code_linux_qa.py",
        "total_examples": len(examples),
        "category_counts": dict(sorted(categories.items())),
        "language_counts": dict(sorted(languages.items())),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{dataset}: {len(examples)} examples -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sources",
    )
    args = parser.parse_args()
    examples = all_examples()
    validate(examples)
    for dataset in ("code", "linux"):
        write_dataset(args.output_root, f"{dataset}_qa", [item for item in examples if item.dataset == dataset])


if __name__ == "__main__":
    main()
