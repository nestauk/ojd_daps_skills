"""
Test cases for the multiskill rules.
"""

import pytest
import spacy

from ojd_daps_skills.extract_skills.multiskill_rules import (
    _split_duplicate_object,
    _split_duplicate_verb,
    _split_on_and,
    _split_skill_mentions,
)


@pytest.fixture
def nlp():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


@pytest.fixture
def duplicate_object_doc(nlp):
    return nlp("using and developing machine learning models")


@pytest.fixture
def duplicate_verb_doc(nlp):
    return nlp("using smartphones and apps")


@pytest.fixture
def skill_mentions_doc(nlp):
    return nlp("written and oral communication skills")


@pytest.fixture
def split_on_and_doc():
    return "machine learning and deep learning"


def test_split_duplicate_object(duplicate_object_doc):
    split_obj = _split_duplicate_object(duplicate_object_doc)
    assert split_obj == [
        "using machine learning models",
        "developing machine learning models",
    ]
    assert len(split_obj) == 2
    assert all(isinstance(skill, str) for skill in split_obj)


def test_split_duplicate_verb(duplicate_verb_doc):
    split_verb = _split_duplicate_verb(duplicate_verb_doc)
    assert split_verb == ["using smartphones", "using apps"]
    assert len(split_verb) == 2
    assert all(isinstance(skill, str) for skill in split_verb)


def test_split_skill_mentions(skill_mentions_doc):
    split_skills = _split_skill_mentions(skill_mentions_doc)
    assert split_skills == ["written skills", "oral communication skills"]
    assert len(split_skills) == 2
    assert all(isinstance(skill, str) for skill in split_skills)
    assert all("skills" in skill for skill in split_skills)


def test_split_on_and(split_on_and_doc):
    split_phrase = _split_on_and(split_on_and_doc)

    assert split_phrase == ["machine learning", "deep learning"]
    assert len(split_phrase) == 2
    assert split_phrase[0] == "machine learning"
    assert split_phrase[1] == "deep learning"
