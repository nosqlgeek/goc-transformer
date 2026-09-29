class WordTokenizer:
    """Split text on whitespace, preserving case and attached punctuation."""

    def tokenize(self, text: str) -> list[str]:
        """Return word tokens; empty or whitespace-only text returns an empty list."""
        return text.split()
