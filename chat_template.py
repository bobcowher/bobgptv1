"""The one place chat messages become model text.

Data builds (scripts/build_mix.py) and, later, the serving code must both use
this, so the model is always prompted in the format it was trained on.
"""

ROLE_HEADERS = {
    "system": "### System",
    "user": "### Question",
    "assistant": "### Answer",
}
END_MARKER = "### End"


def render(messages):
    """A complete conversation, as it appears in training data."""
    return render_with_spans(messages)[0]


def render_with_spans(messages):
    """render(), plus the (start, end) character spans the model should learn to write.

    Each span is an assistant reply plus the separator after it, so the model
    learns where to stop; the last one includes END_MARKER.
    """
    text, spans = "", []
    for i, m in enumerate(messages):
        if i:
            text += "\n\n"
        text += f"{ROLE_HEADERS[m['role']]}\n"
        start = len(text)
        text += m["content"].strip()
        if m["role"] == "assistant":
            spans.append((start, len(text) + 2))  # + the "\n\n" before the next header
    text += f"\n\n{END_MARKER}"
    if spans and spans[-1][1] == len(text) - len(END_MARKER):
        spans[-1] = (spans[-1][0], len(text))
    return text, spans

