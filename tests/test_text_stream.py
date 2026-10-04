import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_stream import TextStream

tokenizer = tiktoken.get_encoding("gpt2")


def stream(text):
    """Feed text through TextStream one token at a time; return the pieces and the stream."""
    ts = TextStream(tokenizer)
    pieces = []
    for token_id in tokenizer.encode(text):
        pieces.append(ts.push(token_id))
        if ts.stopped:
            break
    else:
        pieces.append(ts.finish())
    return pieces, ts


def test_end_marker_is_not_sent():
    pieces, ts = stream(" A dict maps keys to values.\n\n### End")
    assert "".join(pieces) == "A dict maps keys to values."
    assert ts.stopped
    assert ts.finish() == ""


def test_next_question_header_is_not_sent():
    pieces, ts = stream("Use a list.\n\n### Question\nWhat next?")
    assert "".join(pieces) == "Use a list."
    assert ts.stopped


def test_hash_that_is_not_a_marker_is_released():
    pieces, ts = stream("Try C# or # comments.")
    assert "".join(pieces) == "Try C# or # comments."
    assert not ts.stopped


def test_text_is_sent_as_it_arrives():
    pieces, _ = stream("one two three four")
    # Every word but the last goes out before the stream ends.
    assert "".join(pieces[:-1]).startswith("one two three")


def test_split_utf8_character_is_held_until_complete():
    pieces, _ = stream("café ☕ done")
    assert "".join(pieces) == "café ☕ done"
    assert all("�" not in p for p in pieces)


def test_matches_the_block_path_trim():
    text = "  Some code:\n```py\nx = 1\n```\n\n### End\n\n### Question"
    pieces, _ = stream(text)
    assert "".join(pieces) == text[:text.find("### End")].strip()
