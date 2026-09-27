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
    turns = [f"{ROLE_HEADERS[m['role']]}\n{m['content'].strip()}" for m in messages]
    return "\n\n".join(turns) + f"\n\n{END_MARKER}"

