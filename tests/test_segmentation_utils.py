"""
Tests for segmentation.py utility functions:
- normalize_text
- split_into_sentences
- should_discard
"""

import pytest
from pathlib import Path
import sys

# Add the finetune_tts module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune_tts.segmentation import (
    normalize_text,
    split_into_sentences,
    should_discard,
    MIN_DURATION,
    MAX_DURATION,
    MIN_WORDS,
    MIN_SCORE,
)


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_basic_normalization(self):
        """Test basic whitespace normalization."""
        text = "   Silav hevalno   "
        result = normalize_text(text)
        assert result == "Silav hevalno"

    def test_single_spaces_preserved(self):
        """Test that single spaces are preserved."""
        text = "Ramyar diçû"
        result = normalize_text(text)
        assert result == "Ramyar diçû"

    def test_tabs_and_newlines_normalized(self):
        """Test that tabs and newlines are converted to single spaces."""
        text = "Nasa\t\tdîsa\ndiçe heyvê"
        result = normalize_text(text)
        assert result == "Nasa dîsa diçe heyvê"

    def test_mixed_whitespace(self):
        """Test mixed whitespace normalization."""
        text = "  ez  \n\t  diçim  \r  dibistanê  "
        result = normalize_text(text)
        assert result == "ez diçim dibistanê"

    def test_empty_string(self):
        """Test empty string handling."""
        result = normalize_text("")
        assert result == ""

    def test_whitespace_only_string(self):
        """Test whitespace-only string."""
        result = normalize_text("   \t\n  ")
        assert result == ""

    @pytest.mark.parametrize(
        "text_decomposed,expected_normalized",
        [
            (
                "Ez dixwazim bi kurdî biaxivim.", # i + combining ^
                "Ez dixwazim bi kurdî biaxivim.",
            ),
            (
                "Rojbaş, em ê herin şaredariyê.",  # e + combining ^
                "Rojbaş, em ê herin şaredariyê.",
            ),
        ],
    )
    def test_unicode_normalization(self, text_decomposed, expected_normalized):
        """Test Unicode NFC normalization with Kurdish diacritics.

        Tests that both precomposed characters (î, ê, û) and decomposed forms
        (i + ˆ, e + ˆ, u + ˆ) are properly normalized to the same canonical form.
        """
        assert text_decomposed != expected_normalized
        result = normalize_text(text_decomposed)
        assert result == expected_normalized
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_modification_needed(self):
        """Test text that doesn't need modification."""
        text = "Ez dikim biçim mala xwe"
        result = normalize_text(text)
        assert result == "Ez dikim biçim mala xwe"


class TestSplitIntoSentences:
    """Tests for split_into_sentences function."""

    def test_single_sentence_period(self):
        """Test single sentence ending with period."""
        text = "Ew diçû mektebê."
        result = split_into_sentences(text)
        assert result == ["Ew diçû mektebê."]

    def test_single_sentence_exclamation(self):
        """Test single sentence ending with exclamation."""
        text = "Ev pir xweş e!"
        result = split_into_sentences(text)
        assert result == ["Ev pir xweş e!"]

    def test_single_sentence_question(self):
        """Test single sentence ending with question mark."""
        text = "Tu kî yî?"
        result = split_into_sentences(text)
        assert result == ["Tu kî yî?"]

    def test_multiple_sentences(self):
        """Test multiple sentences."""
        text = "Ramyar diçû derve. Êvara we bixêr! Tu çawa yî?"
        result = split_into_sentences(text)
        assert len(result) == 3
        assert result[0] == "Ramyar diçû derve."
        assert result[1] == "Êvara we bixêr!"
        assert result[2] == "Tu çawa yî?"

    def test_multiple_punctuation_marks(self):
        """Test sentences with multiple punctuation marks."""
        text = "Wow!! Rastî?? Belê!!!"
        result = split_into_sentences(text)
        assert len(result) == 3
        assert result[0] == "Wow!!"
        assert result[1] == "Rastî??"
        assert result[2] == "Belê!!!"

    def test_empty_string(self):
        """Test empty string."""
        result = split_into_sentences("")
        assert result == []

    def test_no_punctuation(self):
        """Test text with no sentence-ending punctuation."""
        text = "Ez ditirsim te nebînim"
        result = split_into_sentences(text)
        assert result == []

    def test_leading_trailing_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        text = "   Yekem.   Duyem.   "
        result = split_into_sentences(text)
        assert result == ["Yekem.", "Duyem."]
        # Verify no extra whitespace
        for sentence in result:
            assert sentence == sentence.strip()

    def test_multiple_spaces_between_sentences(self):
        """Test multiple spaces between sentences."""
        text = "Yekem.    Duyem!"
        result = split_into_sentences(text)
        assert len(result) == 2

    def test_sentence_with_special_characters(self):
        """Test sentences with special characters."""
        text = "Salav (cîhan)! Nivîsa baş [nûçe]?"
        result = split_into_sentences(text)
        assert len(result) == 2
        assert result[0] == "Salav (cîhan)!"
        assert result[1] == "Nivîsa baş [nûçe]?"

    def test_mixed_punctuation(self):
        """Test mixed punctuation types."""
        text = "Dest pê bike! Lawîn tê. Ev kî ye?"
        result = split_into_sentences(text)
        assert len(result) == 3

    def test_abbreviation(self):
        text = "Prof. Dr. Smith diçe fezayê."
        result = split_into_sentences(text)
        assert len(result) == 3
        # it is okay to split on the period after "Dr." since there are a few abbreviations in Kurdish that end with a period
        # The should_discard function discards short sentences that are likely to be abbreviations, so we don't need to worry about that here
        assert result[0] == "Prof."
        assert result[1] == "Dr."
        assert result[2] == "Smith diçe fezayê."


class TestShouldDiscard:
    """Tests for should_discard function."""
    def test_digit_detection(self):
        """Test digit detection."""
        sentence = "Sala 2024 pir zehmet bû"  # Contains digit
        duration = 5.0  # Valid duration
        score = 0.0  # Good score
        should_discard_result, reason = should_discard(sentence, duration, score)
        assert should_discard_result is True
        assert reason == "digits"

    def test_abbreviation(self):
        """Test Kurdish abbreviations."""
        sentence = "PDK û PYD du partîyên siyasî yên Kurdan ne."
        duration = 5.0
        score = 0.0
        should_discard_result, reason = should_discard(sentence, duration, score)
        assert should_discard_result is True
        assert reason == "abbreviations"

    def test_dots_abbreviation(self):
        """Test Kurdish text with dotted abbreviations."""
        sentence = "D.Y.A navê welatekî ye"
        duration = 5.0
        score = 0.0
        should_discard_result, reason = should_discard(sentence, duration, score)
        assert should_discard_result is True
        assert reason == "abbreviations"
