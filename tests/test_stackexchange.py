import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

py7zr = pytest.importorskip("py7zr")

from prepare_stackexchange import collect, html_to_text, parse_tags, render


def test_html_to_text_keeps_code_and_lists():
    html = (
        "<p>Use <code>grep -r</code> like   this:</p>\n\n"
        "<pre><code>grep -r &quot;foo&quot; .\n  indented\n</code></pre>\n\n"
        "<ul>\n<li>one</li>\n<li>two &amp; three</li>\n</ul>\n"
    )
    assert html_to_text(html) == (
        "Use `grep -r` like this:\n\n"
        "```\ngrep -r \"foo\" .\n  indented\n```\n\n"
        "- one\n- two & three"
    )


def test_parse_tags_both_formats():
    assert parse_tags("|python|pandas|") == ["python", "pandas"]
    assert parse_tags("<python><pandas>") == ["python", "pandas"]


def test_collect_filters_and_orders(tmp_path):
    posts = tmp_path / "Posts.xml"
    posts.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<posts>\n'
        '<row Id="1" PostTypeId="1" Score="5" AcceptedAnswerId="3" Title="Q one" Tags="|bash|" Body="&lt;p&gt;q&lt;/p&gt;" />\n'
        '<row Id="2" PostTypeId="1" Score="5" ClosedDate="2020-01-01" Title="closed" Tags="|bash|" Body="x" />\n'
        '<row Id="3" PostTypeId="2" ParentId="1" Score="0" Body="&lt;p&gt;accepted&lt;/p&gt;" />\n'
        '<row Id="4" PostTypeId="2" ParentId="1" Score="9" Body="&lt;p&gt;top&lt;/p&gt;" />\n'
        '<row Id="5" PostTypeId="2" ParentId="1" Score="0" Body="&lt;p&gt;dropped&lt;/p&gt;" />\n'
        '<row Id="6" PostTypeId="1" Score="0" Title="low" Tags="|bash|" Body="x" />\n'
        '<row Id="7" PostTypeId="1" Score="3" Title="Off topic" Tags="|java|" Body="x" />\n'
        '</posts>\n'
    )
    assert set(collect(posts, "unix.stackexchange.com", 1, 3)) == {1, 7}
    questions = collect(posts, "stackoverflow.com", 1, 3)
    assert set(questions) == {1}
    assert render(questions[1]) == (
        "# Q one\n\nTags: bash\n\nq\n\n"
        "## Accepted answer (score 0)\n\naccepted\n\n"
        "## Answer (score 9)\n\ntop"
    )
