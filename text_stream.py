from chat_template import END_MARKER

# The model ends an answer with "### End"; if it skips that and starts a new
# turn, the "\n### Question" header is where the answer stops.
STOP_STRINGS = (END_MARKER, "\n### Question")


class TextStream:
    """Turns generated token ids into text that is safe to send to a client.

    push() takes one token id and returns the new text that can be sent now,
    which may be "". It holds back:
      - any tail that could be the start of a stop string ("#", "###", "\\n"),
        until the next token shows whether it is one,
      - trailing whitespace, so the "\\n\\n" before "### End" is never sent,
      - a token that ends partway through a UTF-8 character (decode gives
        "\\ufffd"), until the token that completes it arrives.
    Once a stop string appears, `stopped` is True and everything from the
    stop string on is dropped. finish() returns whatever is still held.
    """

    def __init__(self, tokenizer, stop_strings=STOP_STRINGS):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.token_ids = []
        self.text = ""      # everything decoded so far, leading whitespace removed
        self.sent = 0       # how many characters of self.text have been returned
        self.stopped = False

    def push(self, token_id):
        self.token_ids.append(token_id)
        # Decode the whole reply each time: a single token can be half a character.
        text = self.tokenizer.decode(self.token_ids)
        if text.endswith("�"):
            return ""
        self.text = text.lstrip()

        cuts = [i for i in (self.text.find(s) for s in self.stop_strings) if i != -1]
        if cuts:
            self.stopped = True
            return self._take(min(cuts))

        return self._take(len(self.text) - self._partial_stop_length())

    def finish(self):
        """The held-back text once generation has ended (trailing whitespace dropped)."""
        if self.stopped:
            return ""
        return self._take(len(self.text))

    def _partial_stop_length(self):
        # Length of the longest tail of self.text that is a prefix of a stop string.
        longest = 0
        for s in self.stop_strings:
            for k in range(min(len(s) - 1, len(self.text)), longest, -1):
                if self.text.endswith(s[:k]):
                    longest = k
                    break
        return longest

    def _take(self, end):
        # Send up to `end`, minus trailing whitespace, which waits for the next push.
        end = self.sent + len(self.text[self.sent:end].rstrip())
        piece = self.text[self.sent:end]
        self.sent = max(self.sent, end)
        return piece
