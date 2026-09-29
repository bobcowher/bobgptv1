#!/usr/bin/env python3
"""Generate a large, deterministic code and Linux instruction curriculum.

This complements the smaller hand-authored sets with many executable-style
worked problems.  Diversity comes from multiple reasoning families, data
shapes, APIs, edge cases, and language idioms--not mix-level duplication.

Outputs:
  data/sources/code_curriculum/code_curriculum.jsonl
  data/sources/linux_curriculum/linux_curriculum.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CODE_SYSTEM = (
    "You are a precise programming tutor. Work through the concrete problem, "
    "give a correct answer, explain the key invariant or failure mode, and use "
    "idiomatic code for the requested language."
)
LINUX_SYSTEM = (
    "You are a cautious Linux systems tutor. Give an inspect-first answer, "
    "explain each relevant option, preserve quoting, identify privilege or "
    "namespace assumptions, and warn before destructive operations."
)


@dataclass(frozen=True)
class Example:
    dataset: str
    language: str
    category: str
    family: str
    question: str
    answer: str


def fence(language: str, code: str) -> str:
    return f"```{language}\n{code.rstrip()}\n```"


def take_product(limit: int, *values):
    return itertools.islice(itertools.product(*values), limit)


def language_name(language: str) -> str:
    return {"python": "Python", "rust": "Rust", "cpp": "C++20"}[language]


def list_literal(language: str, values: list[int]) -> str:
    body = ", ".join(map(str, values))
    if language == "python":
        return f"[{body}]"
    if language == "rust":
        return f"vec![{body}]"
    return f"std::vector<int>{{{body}}}"


def trace_accumulators() -> list[Example]:
    examples = []
    combinations = take_product(
        720,
        range(-4, 8),
        range(3, 12),
        range(2, 7),
        range(1, 5),
        range(1, 4),
    )
    for case, (seed, stop, divisor, factor, penalty) in enumerate(combinations, 1):
        total = seed
        for value in range(1, stop):
            if value % divisor == 0:
                total += value * factor
            else:
                total -= penalty
        codes = {
            "python": (
                f"total = {seed}\nfor value in range(1, {stop}):\n"
                f"    if value % {divisor} == 0:\n        total += value * {factor}\n"
                f"    else:\n        total -= {penalty}\nprint(total)"
            ),
            "rust": (
                f"fn main() {{\n    let mut total = {seed};\n    for value in 1..{stop} {{\n"
                f"        if value % {divisor} == 0 {{ total += value * {factor}; }}\n"
                f"        else {{ total -= {penalty}; }}\n    }}\n    println!(\"{{total}}\");\n}}"
            ),
            "cpp": (
                f"#include <iostream>\n\nint main() {{\n    int total = {seed};\n"
                f"    for (int value = 1; value < {stop}; ++value) {{\n"
                f"        if (value % {divisor} == 0) total += value * {factor};\n"
                f"        else total -= {penalty};\n    }}\n    std::cout << total << '\\n';\n}}"
            ),
        }
        for language, code in codes.items():
            question = f"What exact integer does this {language_name(language)} program print?\n\n{fence(language, code)}"
            answer = (
                f"It prints `{total}`. The loop visits the half-open range 1 through {stop - 1}; "
                f"multiples of {divisor} add `value * {factor}`, while every other iteration subtracts {penalty}."
            )
            examples.append(Example("code", language, "trace", "conditional_accumulator", question, answer))
    return examples


def trace_recurrences() -> list[Example]:
    examples = []
    combinations = take_product(480, range(-3, 8), range(1, 6), range(2, 9), range(2, 6))
    for first, increment, modulus, count in combinations:
        values = [first]
        for index in range(1, count):
            values.append((values[-1] * 2 + increment + index) % modulus)
        expected = " ".join(map(str, values))
        codes = {
            "python": (
                f"values = [{first}]\nfor index in range(1, {count}):\n"
                f"    values.append((values[-1] * 2 + {increment} + index) % {modulus})\n"
                "print(*values)"
            ),
            "rust": (
                f"fn main() {{\n    let mut values = vec![{first}];\n    for index in 1..{count} {{\n"
                f"        let next = (values[values.len() - 1] * 2 + {increment} + index as i32).rem_euclid({modulus});\n"
                "        values.push(next);\n    }\n    let output: Vec<String> = values.iter().map(ToString::to_string).collect();\n"
                "    println!(\"{}\", output.join(\" \"));\n}"
            ),
            "cpp": (
                f"#include <iostream>\n#include <vector>\n\nint main() {{\n    std::vector<int> values{{{first}}};\n"
                f"    for (int index = 1; index < {count}; ++index) {{\n"
                f"        const int raw = values.back() * 2 + {increment} + index;\n"
                f"        values.push_back((raw % {modulus} + {modulus}) % {modulus});\n    }}\n"
                "    for (std::size_t i = 0; i < values.size(); ++i) {\n"
                "        if (i) std::cout << ' ';\n        std::cout << values[i];\n    }\n    std::cout << '\\n';\n}"
            ),
        }
        for language, code in codes.items():
            answer = (
                f"The exact output is `{expected}`. Each new element uses the element just appended, "
                "then the modulus folds the result into a bounded range."
            )
            examples.append(
                Example(
                    "code",
                    language,
                    "trace",
                    "state_recurrence",
                    f"Trace this {language_name(language)} recurrence. What line is printed?\n\n{fence(language, code)}",
                    answer,
                )
            )
    return examples


def trace_collections() -> list[Example]:
    examples = []
    combinations = take_product(540, range(-5, 4), range(5, 11), range(2, 7), range(0, 3), range(1, 5))
    for start, length, divisor, remainder, offset in combinations:
        values = list(range(start, start + length))
        selected = [value * value + offset for value in values if value % divisor == remainder]
        rendered = "[" + ", ".join(map(str, selected)) + "]"
        literals = {language: list_literal(language, values) for language in ("python", "rust", "cpp")}
        codes = {
            "python": (
                f"values = {literals['python']}\n"
                f"result = [value * value + {offset} for value in values if value % {divisor} == {remainder}]\n"
                "print(result)"
            ),
            "rust": (
                f"fn main() {{\n    let values: Vec<i32> = {literals['rust']};\n"
                f"    let result: Vec<i32> = values.into_iter().filter(|value| (*value).rem_euclid({divisor}) == {remainder})"
                f".map(|value| value * value + {offset}).collect();\n"
                "    println!(\"{:?}\", result);\n}"
            ),
            "cpp": (
                "#include <iostream>\n#include <vector>\n\nint main() {\n"
                f"    auto values = {literals['cpp']};\n    std::vector<int> result;\n"
                f"    for (int value : values) if (((value % {divisor}) + {divisor}) % {divisor} == {remainder}) "
                f"result.push_back(value * value + {offset});\n"
                "    std::cout << '[';\n    for (std::size_t i = 0; i < result.size(); ++i) {\n"
                "        if (i) std::cout << \", \";\n        std::cout << result[i];\n    }\n    std::cout << \"]\\n\";\n}"
            ),
        }
        for language, code in codes.items():
            question = f"Determine the exact list printed by this {language_name(language)} code.\n\n{fence(language, code)}"
            answer = (
                f"It prints `{rendered}`. Filtering happens before the square-and-offset mapping. "
                f"The remainder check uses mathematical nonnegative remainders, including for negative inputs."
            )
            examples.append(Example("code", language, "trace", "filter_map", question, answer))
    return examples


def threshold_implementations() -> list[Example]:
    examples = []
    combinations = take_product(
        540,
        ("temperatures", "scores", "samples", "measurements", "balances", "latencies"),
        range(-12, 18),
        ("strictly above", "at least"),
        ("double", "absolute", "offset"),
    )
    for noun, threshold, comparison, transform in combinations:
        suffix = hashlib.sha256(f"{noun}:{threshold}:{comparison}:{transform}".encode()).hexdigest()[:8]
        function = f"select_{noun}_{suffix}"
        operator = ">" if comparison == "strictly above" else ">="
        if transform == "double":
            py_expr, rust_expr, cpp_expr, description = "value * 2", "value * 2", "value * 2", "double each retained value"
        elif transform == "absolute":
            py_expr, rust_expr, cpp_expr, description = "abs(value)", "value.abs()", "std::abs(value)", "take each retained value's absolute value"
        else:
            py_expr = rust_expr = cpp_expr = f"value + {abs(threshold) + 1}"
            description = f"add {abs(threshold) + 1} to each retained value"
        codes = {
            "python": (
                f"def {function}(values: list[int]) -> list[int]:\n"
                f"    return [{py_expr} for value in values if value {operator} {threshold}]"
            ),
            "rust": (
                f"fn {function}(values: &[i32]) -> Vec<i32> {{\n"
                f"    values.iter().copied().filter(|value| *value {operator} {threshold})"
                f".map(|value| {rust_expr}).collect()\n}}"
            ),
            "cpp": (
                "#include <cstdlib>\n#include <span>\n#include <vector>\n\n"
                f"std::vector<int> {function}(std::span<const int> values) {{\n    std::vector<int> result;\n"
                f"    for (int value : values) if (value {operator} {threshold}) result.push_back({cpp_expr});\n"
                "    return result;\n}"
            ),
        }
        prompt = (
            f"Write `{function}` to retain {noun} {comparison} {threshold}, preserve their order, and {description}. "
            "The input must not be mutated."
        )
        for language, code in codes.items():
            answer = (
                "Make filtering and transformation explicit in one pass. The result owns its values, and the input is only read.\n\n"
                + fence(language, code)
            )
            examples.append(
                Example("code", language, "implementation", "filtered_transform", f"In {language_name(language)}, {prompt}", answer)
            )
    return examples


def window_implementations() -> list[Example]:
    examples = []
    combinations = take_product(
        420,
        range(2, 9),
        ("sum", "maximum", "minimum"),
        ("readings", "samples", "values", "measurements", "deltas"),
        ("reject", "empty"),
    )
    for width, operation, noun, invalid_policy in combinations:
        suffix = hashlib.sha256(f"{width}:{operation}:{noun}:{invalid_policy}".encode()).hexdigest()[:8]
        function = f"windows_{operation}_{suffix}"
        prompt = (
            f"Write `{function}` that returns the {operation} of every contiguous window of width {width} in integer {noun}. "
            + ("Raise or return an error when the input is shorter than the window." if invalid_policy == "reject" else "Return an empty result when the input is shorter than the window.")
        )
        if operation == "sum":
            py_calc = "sum(window)"
            rust_calc = "window.iter().sum()"
            cpp_calc = "std::accumulate(window.begin(), window.end(), 0)"
            cpp_header = "#include <numeric>\n"
        elif operation == "maximum":
            py_calc = "max(window)"
            rust_calc = "*window.iter().max().unwrap()"
            cpp_calc = "*std::max_element(window.begin(), window.end())"
            cpp_header = "#include <algorithm>\n"
        else:
            py_calc = "min(window)"
            rust_calc = "*window.iter().min().unwrap()"
            cpp_calc = "*std::min_element(window.begin(), window.end())"
            cpp_header = "#include <algorithm>\n"
        py_guard = "raise ValueError('input shorter than window')" if invalid_policy == "reject" else "return []"
        rust_return = "Result<Vec<i32>, &'static str>" if invalid_policy == "reject" else "Vec<i32>"
        rust_guard = "return Err(\"input shorter than window\");" if invalid_policy == "reject" else "return Vec::new();"
        rust_result = "Ok(result)" if invalid_policy == "reject" else "result"
        cpp_return = "std::optional<std::vector<int>>" if invalid_policy == "reject" else "std::vector<int>"
        cpp_guard = "return std::nullopt;" if invalid_policy == "reject" else "return {};"
        cpp_result = "return result;"
        codes = {
            "python": (
                f"def {function}(values: list[int]) -> list[int]:\n    width = {width}\n"
                f"    if len(values) < width:\n        {py_guard}\n"
                f"    return [{py_calc} for start in range(len(values) - width + 1) "
                "for window in [values[start:start + width]]]"
            ),
            "rust": (
                f"fn {function}(values: &[i32]) -> {rust_return} {{\n    const WIDTH: usize = {width};\n"
                f"    if values.len() < WIDTH {{ {rust_guard} }}\n"
                f"    let result = values.windows(WIDTH).map(|window| {rust_calc}).collect();\n    {rust_result}\n}}"
            ),
            "cpp": (
                f"{cpp_header}#include <optional>\n#include <span>\n#include <vector>\n\n"
                f"{cpp_return} {function}(std::span<const int> values) {{\n    constexpr std::size_t width = {width};\n"
                f"    if (values.size() < width) {cpp_guard}\n    std::vector<int> result;\n"
                "    for (std::size_t start = 0; start + width <= values.size(); ++start) {\n"
                "        auto window = values.subspan(start, width);\n"
                f"        result.push_back({cpp_calc});\n    }}\n    {cpp_result}\n}}"
            ),
        }
        for language, code in codes.items():
            answer = (
                "A window view/slice makes the boundaries explicit, including the final valid starting position. "
                "The short-input behavior follows the requested contract.\n\n" + fence(language, code)
            )
            examples.append(Example("code", language, "implementation", "fixed_windows", f"In {language_name(language)}, {prompt}", answer))
    return examples


def validation_implementations() -> list[Example]:
    examples = []
    combinations = take_product(
        360,
        ("username", "tag", "identifier", "slug", "label", "key"),
        range(3, 9),
        range(10, 25),
        ("underscore", "hyphen", "both"),
    )
    for noun, minimum, maximum, separators in combinations:
        allowed = "_" if separators == "underscore" else "-" if separators == "hyphen" else "_-"
        suffix = hashlib.sha256(f"{noun}:{minimum}:{maximum}:{allowed}".encode()).hexdigest()[:8]
        function = f"valid_{noun}_{suffix}"
        prompt = (
            f"Write `{function}` for an ASCII {noun}. Its length must be between {minimum} and {maximum}; "
            f"the first character must be a lowercase letter; later characters may be lowercase letters, digits, or `{allowed}`."
        )
        codes = {
            "python": (
                f"def {function}(text: str) -> bool:\n"
                f"    if not ({minimum} <= len(text) <= {maximum}) or not ('a' <= text[0] <= 'z'):\n        return False\n"
                f"    return all(('a' <= char <= 'z') or char.isascii() and char.isdigit() or char in {allowed!r} for char in text[1:])"
            ),
            "rust": (
                f"fn {function}(text: &str) -> bool {{\n    let bytes = text.as_bytes();\n"
                f"    if !({minimum}..={maximum}).contains(&bytes.len()) || !bytes[0].is_ascii_lowercase() {{ return false; }}\n"
                f"    bytes[1..].iter().all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || {allowed.encode()!r}.contains(byte))\n}}"
            ).replace("b'_-'", "b\"_-\"").replace("b'_'", "b\"_\"").replace("b'-'", "b\"-\""),
            "cpp": (
                "#include <cctype>\n#include <string_view>\n\n"
                f"bool {function}(std::string_view text) {{\n"
                f"    if (text.size() < {minimum} || text.size() > {maximum} || text.empty() || !(text[0] >= 'a' && text[0] <= 'z')) return false;\n"
                "    for (unsigned char ch : text.substr(1)) {\n"
                f"        if (!((ch >= 'a' && ch <= 'z') || std::isdigit(ch) || std::string_view({allowed!r}).find(ch) != std::string_view::npos)) return false;\n"
                "    }\n    return true;\n}"
            ).replace("std::string_view('_-')", 'std::string_view("_-")').replace("std::string_view('_')", 'std::string_view("_")').replace("std::string_view('-')", 'std::string_view("-")'),
        }
        for language, code in codes.items():
            answer = (
                "Check length and the special first-character rule before scanning the remainder. "
                "The implementation deliberately uses ASCII rules rather than locale- or Unicode-dependent categories.\n\n"
                + fence(language, code)
            )
            examples.append(Example("code", language, "implementation", "ascii_validation", f"In {language_name(language)}, {prompt}", answer))
    return examples


def debugging_boundaries() -> list[Example]:
    examples = []
    combinations = take_product(
        300,
        ("items", "values", "records", "samples", "events"),
        range(2, 12),
        ("include_end", "exclude_end", "avoid_index"),
        ("empty", "singleton", "last"),
    )
    for noun, size, failure, edge in combinations:
        if failure == "include_end":
            bug = f"iterates through index `{size}` for a collection of length `{size}`"
            fix = f"use the half-open range from zero up to, but not including, `{size}`"
        elif failure == "exclude_end":
            bug = "stops one position before the final valid element"
            fix = "make the stop bound the collection length, which remains exclusive"
        else:
            bug = "uses indexing even though only each value is needed, creating an unnecessary boundary hazard"
            fix = "iterate directly over the collection"
        question = (
            f"A loop over `{noun}` {bug}. The failing test is the `{edge}` case. "
            f"Explain the boundary invariant and the minimal repair for a collection length of {size}."
        )
        answer = (
            f"Valid indices are `0` through `{size - 1}`. The repair is to {fix}. "
            "Tests should cover empty input, one element, and the final element because those expose incorrect inclusive/exclusive assumptions."
        )
        for language in ("python", "rust", "cpp"):
            examples.append(Example("code", language, "debugging", "boundary_invariant", f"In {language_name(language)}: {question}", answer))
    return examples


def test_design() -> list[Example]:
    examples = []
    combinations = take_product(
        300,
        ("deduplicate", "lower_bound", "parse_record", "merge_ranges", "moving_average", "normalize_name"),
        ("empty input", "single element", "duplicate boundary", "invalid syntax", "largest input", "negative values"),
        ("unit", "property", "regression"),
        ("mutation", "ordering", "error", "overflow"),
    )
    for function, edge, test_kind, risk in combinations:
        question = (
            f"Design one high-value {test_kind} test for `{function}` focused on {edge} and the risk of {risk}. "
            "State the input, expected observation, and why this test earns its place."
        )
        answer = (
            f"Choose the smallest input that exhibits {edge}, preserve a copy when {risk} could affect input state, "
            f"call `{function}`, and assert both the documented result/error and the relevant {risk} invariant. "
            "The test should fail for the specific plausible defect, not merely execute the branch; avoid asserting unrelated formatting or implementation order."
        )
        for language in ("python", "rust", "cpp"):
            examples.append(Example("code", language, "testing", "edge_case_design", f"For {language_name(language)}, {question}", answer))
    return examples


def linux_find_examples() -> list[Example]:
    examples = []
    roots = ("/var/log", "/srv/app", "/opt/data", "/home/alex/projects", "/tmp/build-cache")
    suffixes = ("log", "json", "tmp", "py", "rs", "cpp", "gz", "csv")
    days = (1, 3, 7, 14, 30, 60)
    sizes = ("1M", "10M", "50M", "100M", "500M")
    for root, suffix, age, size in take_product(720, roots, suffixes, days, sizes):
        command = f"find {json.dumps(root)} -type f -name '*.{suffix}' -mtime +{age} -size +{size} -print"
        question = (
            f"Without changing anything, list regular `.{suffix}` files under `{root}` that are older than {age} days "
            f"and larger than {size}. Explain how to review this safely before any cleanup."
        )
        answer = (
            "Use an observation-only `find` first:\n\n" + fence("bash", command) +
            f"\n\n`-mtime +{age}` uses completed 24-hour periods and `-size +{size}` selects larger files. "
            "Review ownership, active writers, retention policy, mount boundaries, and a representative sample before designing a separate deletion step."
        )
        examples.append(Example("linux", "shell", "filesystem", "find_observe", question, answer))
    return examples


def linux_journal_examples() -> list[Example]:
    examples = []
    services = ("nginx", "sshd", "postgresql", "docker", "api-worker", "backup", "systemd-resolved", "cron")
    boots = (0, -1, -2)
    priorities = ("err", "warning", "notice", "info")
    windows = ("10 min ago", "1 hour ago", "today", "yesterday")
    for service, boot, priority, since in take_product(720, services, boots, priorities, windows):
        command = f"journalctl -u {service}.service -b {boot} -p {priority} --since {json.dumps(since)} --no-pager"
        question = (
            f"Show `{service}.service` journal entries from boot {boot}, priority `{priority}` or more severe, "
            f"since `{since}`, without entering a pager."
        )
        answer = (
            fence("bash", command) +
            f"\n\n`-u` filters the unit, `-b {boot}` selects the boot, `-p {priority}` applies the severity ceiling, "
            "and `--since` applies the time boundary. An empty result can mean no matching entries, unavailable persistent history, or insufficient access."
        )
        examples.append(Example("linux", "shell", "logs", "journal_query", question, answer))
    return examples


def linux_socket_examples() -> list[Example]:
    examples = []
    ports = (22, 53, 80, 443, 5432, 6379, 8000, 8080, 8443, 9090)
    protocols = ("tcp", "udp")
    families = ("any", "ipv4", "ipv6")
    modes = ("listening", "all")
    for port, protocol, family, mode in take_product(360, ports, protocols, families, modes):
        flags = "-l" if mode == "listening" else "-a"
        flags += "t" if protocol == "tcp" else "u"
        if family == "ipv4": flags += "4"
        elif family == "ipv6": flags += "6"
        flags += "np"
        command = f"ss {flags} 'sport = :{port}'"
        question = f"Inspect {mode} {family} {protocol.upper()} sockets using local port {port}, including numeric addresses and owning processes."
        answer = (
            fence("bash", command) +
            "\n\nThis is observational. Process details for sockets owned by other users can require privilege, and containers may place the relevant socket in another network namespace."
        )
        examples.append(Example("linux", "shell", "networking", "socket_inspection", question, answer))
    return examples


def linux_permission_examples() -> list[Example]:
    examples = []
    file_types = ("regular file", "directory")
    owner_modes = range(0, 8)
    group_modes = range(0, 8)
    other_modes = range(0, 8)
    for kind, owner, group, other in take_product(768, file_types, owner_modes, group_modes, other_modes):
        def decode(value: int) -> str:
            return ("r" if value & 4 else "-") + ("w" if value & 2 else "-") + ("x" if value & 1 else "-")
        symbolic = decode(owner) + decode(group) + decode(other)
        mode = f"{owner}{group}{other}"
        if kind == "regular file":
            semantics = "On a regular file, read accesses contents, write changes contents, and execute permits execution subject to other controls."
        else:
            semantics = "On a directory, read lists names, write changes entries, and execute searches/traverses names; useful operations often require combinations."
        question = f"Decode mode `{mode}` on a {kind}. Give the symbolic bits and explain the practical access rather than only reciting digits."
        answer = f"The nine permission bits are `{symbolic}`: owner `{decode(owner)}`, group `{decode(group)}`, others `{decode(other)}`. {semantics}"
        examples.append(Example("linux", "permissions", "permissions", "mode_decode", question, answer))
    return examples


def linux_capability_examples() -> list[Example]:
    capabilities = (
        ("CAP_NET_BIND_SERVICE", "bind an Internet-domain socket to a port below 1024"),
        ("CAP_NET_ADMIN", "configure interfaces, routes, firewall rules, or traffic control in the governed network namespace"),
        ("CAP_CHOWN", "make arbitrary changes to file UID and GID ownership subject to filesystem constraints"),
        ("CAP_DAC_OVERRIDE", "bypass many discretionary file read, write, and execute permission checks"),
        ("CAP_DAC_READ_SEARCH", "bypass discretionary file read and directory search checks without granting general writes"),
        ("CAP_KILL", "bypass UID-based permission checks when sending signals"),
        ("CAP_SETUID", "make permitted manipulations of process user IDs and related credential operations"),
        ("CAP_SETGID", "make permitted manipulations of process group IDs and supplementary groups"),
        ("CAP_SYS_CHROOT", "call chroot"),
        ("CAP_SYS_PTRACE", "trace or inspect otherwise protected processes and use related cross-process interfaces"),
        ("CAP_SYS_TIME", "set the system clock and related time state"),
        ("CAP_MKNOD", "create special files with mknod"),
        ("CAP_AUDIT_WRITE", "write records to the kernel audit log"),
        ("CAP_LEASE", "establish leases on arbitrary files"),
        ("CAP_LINUX_IMMUTABLE", "change immutable and append-only inode flags"),
    )
    contexts = ("systemd service", "container", "maintenance helper", "batch worker", "network daemon")
    sets = ("effective", "permitted", "bounding", "ambient")
    examples = []
    for (capability, operation), context, cap_set in take_product(900, capabilities, contexts, sets):
        question = (
            f"A {context} needs to {operation}. Which Linux capability is the narrow starting point, "
            f"and what must be checked if it appears in the process's {cap_set} set but the operation still fails?"
        )
        answer = (
            f"Start by evaluating `{capability}`, not a catch-all such as `CAP_SYS_ADMIN`. A capability in the {cap_set} set "
            "does not alone prove it is usable: inspect the permitted/effective/bounding/ambient relationship, user and other namespaces, "
            "`no_new_privs`, LSM policy, seccomp, filesystem or mount restrictions, and whether the resource is governed by that namespace. "
            "Confirm the exact failing syscall before granting anything."
        )
        examples.append(Example("linux", "security", "capabilities", "least_capability", question, answer))
    return examples


def linux_shell_safety_examples() -> list[Example]:
    examples = []
    variables = ("path", "output_dir", "archive", "pattern", "service_name", "user_input", "config_file", "mount_point")
    operations = ("printf", "test", "copy", "move", "remove-preview", "checksum")
    hazards = ("spaces", "wildcards", "leading dash", "empty value", "newline")
    for variable, operation, hazard in take_product(600, variables, operations, hazards):
        if operation == "printf": command = f"printf '%s\\n' \"${{{variable}}}\""
        elif operation == "test": command = f"test -e \"${{{variable}}}\""
        elif operation == "copy": command = f"cp -- \"${{{variable}}}\" \"$destination\""
        elif operation == "move": command = f"mv -- \"${{{variable}}}\" \"$destination\""
        elif operation == "remove-preview": command = f"printf 'would remove: %q\\n' \"${{{variable}}}\""
        else: command = f"sha256sum -- \"${{{variable}}}\""
        question = (
            f"Show a safe shell form for `{operation}` using variable `${variable}` when its value may contain {hazard}. "
            "Do not perform deletion in the example."
        )
        answer = (
            fence("bash", command) +
            "\n\nDouble quotes preserve the value as one argument, and `--` ends option parsing where supported so a leading dash is data. "
            "For an operation with serious consequences, reject an empty/unexpected path and preview exact quoted arguments before a separate authorized mutation."
        )
        examples.append(Example("linux", "shell", "shell_safety", "quoted_argument", question, answer))
    return examples


def build_examples() -> list[Example]:
    groups = (
        trace_accumulators(),
        trace_recurrences(),
        trace_collections(),
        threshold_implementations(),
        window_implementations(),
        validation_implementations(),
        debugging_boundaries(),
        test_design(),
        linux_find_examples(),
        linux_journal_examples(),
        linux_socket_examples(),
        linux_permission_examples(),
        linux_capability_examples(),
        linux_shell_safety_examples(),
    )
    return [example for group in groups for example in group]


def validate(examples: list[Example]) -> None:
    pairs = [(item.question, item.answer) for item in examples]
    if len(pairs) != len(set(pairs)):
        counts = Counter(pairs)
        duplicate = next(pair for pair, count in counts.items() if count > 1)
        raise ValueError(f"duplicate Q&A pair: {duplicate[0][:120]}")
    for item in examples:
        if not item.question.strip() or not item.answer.strip():
            raise ValueError(f"empty {item.family} example")
        for code in re.findall(r"```python\n(.*?)```", item.question + item.answer, re.DOTALL):
            try:
                ast.parse(code)
            except SyntaxError as error:
                raise ValueError(f"invalid Python in {item.family}: {error}\n{code}") from error


def as_record(example: Example, index: int) -> dict:
    prefix = "code_curriculum" if example.dataset == "code" else "linux_curriculum"
    return {
        "id": f"{prefix}/{index:06d}",
        "messages": [
            {"role": "system", "content": CODE_SYSTEM if example.dataset == "code" else LINUX_SYSTEM},
            {"role": "user", "content": example.question},
            {"role": "assistant", "content": example.answer},
        ],
        "metadata": {
            "language": example.language,
            "category": example.category,
            "family": example.family,
            "license": "CC0-1.0",
            "origin": "synthetic: deterministic authored templates for llm_bobgpt",
        },
    }


def write_dataset(root: Path, source: str, examples: list[Example]) -> None:
    output_dir = root / source
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [as_record(example, index) for index, example in enumerate(examples, 1)]
    (output_dir / f"{source}.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    metadata = {
        "description": "Large deterministic worked-problem curriculum with no mix-level repetition.",
        "license": "CC0-1.0",
        "generator": "scripts/generate_large_qa.py",
        "total_examples": len(examples),
        "language_counts": dict(sorted(Counter(item.language for item in examples).items())),
        "category_counts": dict(sorted(Counter(item.category for item in examples).items())),
        "family_counts": dict(sorted(Counter(item.family for item in examples).items())),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{source}: {len(examples):,} examples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sources",
    )
    args = parser.parse_args()
    examples = build_examples()
    validate(examples)
    code = [item for item in examples if item.dataset == "code"]
    linux = [item for item in examples if item.dataset == "linux"]
    write_dataset(args.output_root, "code_curriculum", code)
    write_dataset(args.output_root, "linux_curriculum", linux)


if __name__ == "__main__":
    main()
