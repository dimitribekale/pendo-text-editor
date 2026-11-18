"""Cleaners used when building the writing corpus."""

from model_training.data_preprocessing.text_cleaning import (
    detokenize,
    is_usable,
    normalize_whitespace,
    remove_blog_markers,
    strip_news_dateline,
    truncate_at_paragraph,
    unwrap_lines,
)


def test_whitespace_runs_are_collapsed():
    assert normalize_whitespace("  a   b \n\n\n\n c\t d  ") == "a b\n\nc d"


def test_hard_wrapped_lines_become_paragraphs():
    assert unwrap_lines("one\ntwo\n\nthree") == "one two\n\nthree"


def test_detokenize_reattaches_punctuation_and_contractions():
    raw = "She did n't stop , and it 's fine . `` Really ? '' ( yes )"
    assert detokenize(raw) == 'She didn\'t stop, and it\'s fine. "Really?" (yes)'


def test_detokenize_restores_newlines():
    assert detokenize("one <newline> two") == "one \n two"


def test_news_dateline_is_removed():
    assert strip_news_dateline("(CNN)The BBC has") == "The BBC has"
    assert strip_news_dateline("LONDON, England (CNN) -- A man") == "A man"
    assert strip_news_dateline("No marker here") == "No marker here"


def test_blog_link_placeholders_are_removed():
    assert remove_blog_markers("see urlLink this") == "see  this"


def test_long_documents_are_cut_at_a_sentence_end():
    text = "A sentence. " * 2000
    cut = truncate_at_paragraph(text)
    assert len(cut) <= 8000 and cut.endswith(".")


def test_short_or_non_english_documents_are_rejected():
    assert not is_usable("too short")
    assert not is_usable("éèê " * 200)
    assert is_usable("A perfectly ordinary English sentence. " * 10)
