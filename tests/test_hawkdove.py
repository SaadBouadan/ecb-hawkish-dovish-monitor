import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hawkdove import hawkishness, policy_sentences  # noqa: E402


def test_policy_sentences_keeps_only_policy_vocabulary():
    text = ("Inflation is expected to return to our two per cent target over the medium term.[1] "
            "The weather in Frankfurt was pleasant during the conference this week. "
            "Short.")
    out = policy_sentences(text)
    assert out == ["Inflation is expected to return to our two per cent target over the medium term."]


def test_policy_sentences_length_bounds():
    long = "monetary policy " + "word " * 100 + "."
    assert policy_sentences(long) == []
    assert policy_sentences(None) == []


def test_hawkishness_measure():
    s = pd.Series(["hawkish", "hawkish", "dovish", "neutral"])
    assert hawkishness(s) == (2 - 1) / 4
    assert hawkishness(pd.Series([], dtype=str)) == 0.0
    assert hawkishness(pd.Series(["dovish"] * 3)) == -1.0
