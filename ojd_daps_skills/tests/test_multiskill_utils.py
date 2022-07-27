import pytest

from ojd_daps_skills.pipeline.skill_ner.multiskill_utils import (
    duplicate_object,
    duplicate_verb,
    split_skill_mentions,
    split_multiskill,
    split_on_and,
)

import spacy

nlp = spacy.load("en_core_web_sm")


def test_duplicate_object():
    text = "using and providing clinical supervision"
    parsed_text = nlp(text)
    split_text = duplicate_object(parsed_text)

    assert split_text
    assert len(split_text) == 2
    assert split_text[0] == "using clinical supervision"
    assert split_text[1] == "providing clinical supervision"


def test_duplicate_verb():

    text = "using smartphones and apps"
    parsed_text = nlp(text)
    split_text = duplicate_verb(parsed_text)
    assert split_text[0] == "using smartphones"
    assert split_text[1] == "using apps"

    assert split_text
    assert len(split_text) == 2


def test_split_skill_mentions():
    text = "written and oral communication skills"
    parsed_text = nlp(text)
    split_text = split_skill_mentions(parsed_text)

    assert split_text
    assert len(split_text) == 2
    assert split_text[0] == "written skills"
    assert split_text[1] == "oral communication skills"


def test_split_multiskill():
    text = "written and oral communication skills"
    split_text = split_multiskill(text)

    assert split_text
    assert len(split_text) == 2
    assert split_text[0] == "written skills"
    assert split_text[1] == "oral communication skills"


def test_not_split():

    text = "communication skills"
    split_text = split_multiskill(text)

    assert not split_text

    text = "written and oral communication skills"
    split_text = split_multiskill(text, min_length=10)

    assert not split_text


def test_split_on_and():

    text = "a and b"
    split_text = split_on_and(text)

    assert split_text == ["a", "b"]

    text = "a understanding b"
    split_text = split_on_and(text)

    assert split_text == [text]

    text = "a, b, and c"
    split_text = split_on_and(text)

    assert split_text == ["a", "b", "c"]

    text = "a, b, and, c"
    split_text = split_on_and(text)

    assert split_text == ["a", "b", "c"]
