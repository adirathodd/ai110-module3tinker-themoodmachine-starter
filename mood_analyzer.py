# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    def _analyze_text(self, text: str) -> Tuple[int, int, int]:
      """
      Return aggregate sentiment signals for a text.

      Returns:
        (score, positive_hits, negative_hits)
      """
      tokens = self.preprocess(text)
      score = 0
      positive_hits = 0
      negative_hits = 0

      negation_words = {
        "not",
        "no",
        "never",
        "cannot",
        "wasnt",
        "isnt",
        "arent",
        "werent",
        "dont",
        "doesnt",
        "didnt",
        "cant",
        "couldnt",
        "wouldnt",
        "shouldnt",
        "wont",
      }
      positive_extras = {
        "fire",
        "lit",
        "yay",
        "pumped",
        "hopeful",
        "proud",
        "sick",
        "dope",
        "slaps",
        "vibing",
        ":)",
        "😂",
        "mixed"
      }
      negative_extras = {
        "ugh",
        "meh",
        "stuck",
        "disappointed",
        "exhausted",
        "trash",
        "mid",
        "cringe",
        "wack",
        "🥲",
        ":(",
        "💀",
        "mixed"
      }

      i = 0
      while i < len(tokens):
        token = tokens[i]

        # Handle simple negation by flipping the next known sentiment token.
        is_negation = token in negation_words or token.endswith("n't")
        if is_negation and i + 1 < len(tokens):
          nxt = tokens[i + 1]
          if nxt in self.positive_words or nxt in positive_extras:
            score -= 1
            negative_hits += 1
            i += 2
            continue
          if nxt in self.negative_words or nxt in negative_extras:
            score += 1
            positive_hits += 1
            i += 2
            continue

        if token in self.positive_words or token in positive_extras:
          score += 1
          positive_hits += 1
        if token in self.negative_words or token in negative_extras:
          score -= 1
          negative_hits += 1

        i += 1

      return score, positive_hits, negative_hits

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()
        # Collapse long character runs ("soooo" -> "soo") to reduce sparsity.
        cleaned = re.sub(r"(.)\\1{2,}", r"\\1\\1", cleaned)

        # Keep simple words, numbers, and a few high-signal emoji/emoticons.
        token_pattern = r"[a-z]+(?:'[a-z]+)?|\\d+|:\)|:\(|🥲|😂|💀"
        tokens = re.findall(token_pattern, cleaned)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        score, _, _ = self._analyze_text(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score, positive_hits, negative_hits = self._analyze_text(text)
        tokens = self.preprocess(text)

        # Treat weak positivity with uncertainty language as neutral.
        hedging_tokens = {"idk", "just", "weird", "kinda", "kind", "sorta", "maybe"}
        has_hedge = any(token in hedging_tokens for token in tokens)

        # Mixed means we found both positive and negative evidence.
        if positive_hits > 0 and negative_hits > 0:
            return "mixed"
        if score == 1 and negative_hits == 0 and has_hedge:
          return "neutral"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
