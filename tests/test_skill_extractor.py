"""
Test cases for the SkillExtractor class.
"""

import pytest
from spacy.tokens import Doc
from wasabi import msg

from ojd_daps_skills import setup_spacy_extensions
from ojd_daps_skills.extract_skills.extract_skills import SkillsExtractor

setup_spacy_extensions()


@pytest.fixture
def sm():
    return SkillsExtractor(taxonomy_name="toy")


@pytest.fixture
def job_ad():
    return "We are looking for a data scientist with experience in Python, R, and SQL."


@pytest.fixture
def job_ads():
    return [
        "We are looking for a data scientist with experience in Python, R, and SQL.",
        "We are looking for a marketing manager with great oral and written communication skills.",
    ]


@pytest.fixture
def job_doc(sm):
    doc = sm.extract_config.nlp(
        "We are looking for a data scientist with experience in Python, R, and SQL."
    )
    doc._.skill_spans = ["Python", "R", "SQL"]
    return doc


@pytest.fixture
def job_docs(sm):
    docs = [
        sm.extract_config.nlp(
            "We are looking for a data scientist with experience in Python, R, and SQL."
        ),
        sm.extract_config.nlp(
            "We are looking for a marketing manager with great oral and written communication skills."
        ),
    ]

    docs[0]._.skill_spans = ["Python", "R", "SQL"]
    docs[1]._.skill_spans = ["oral communication", "written communication"]

    return docs


def test_extract_skills(sm, job_ads):
    skills_extracted = sm.extract_skills(job_ads)

    # assert its a list
    assert isinstance(skills_extracted, list)
    # assert that its a list of doc objects
    assert all(isinstance(doc, Doc) for doc in skills_extracted)
    # assert that the length of the list is equal to the length of the input
    assert len(skills_extracted) == len(job_ads)
    # assert it has skill_spans attribute
    assert all(doc._.skill_spans for doc in skills_extracted)
    # get attributes


def test_get_skills(sm, job_ad):
    job_skills = sm.get_skills(job_ad)

    assert isinstance(job_skills, Doc)
    assert len(job_skills.ents) == 4
    assert isinstance(job_skills._.skill_spans, list)
    assert len(job_skills._.skill_spans) == 4


def test_map_skills_single(sm, job_doc):
    skills_extracted = sm.map_skills(job_doc)
    msg.info([t._.skill_spans for t in skills_extracted])

    assert isinstance(skills_extracted, list)
    # assert everything is a doc object
    assert all(isinstance(doc, Doc) for doc in skills_extracted)
    # assert that it has mapped_skills attribute
    assert all(doc._.mapped_skills for doc in skills_extracted)


def test_map_skills_multiple(sm, job_docs):
    skills_extracted = sm.map_skills(job_docs)

    assert isinstance(skills_extracted, list)
    # assert everything is a doc object
    assert all(isinstance(doc, Doc) for doc in skills_extracted)
    # assert that it has mapped_skills attribute
    assert all(doc._.mapped_skills for doc in skills_extracted)
    # assert that skills extracted is same len as job ad
    assert len(skills_extracted) == len(job_docs)


def test_call(sm, job_ads):
    job_ads_skills = sm(job_ads)

    assert isinstance(job_ads_skills, list)
    # assert everything is a doc object
    assert all(isinstance(doc, Doc) for doc in job_ads_skills)
    # assert that it has mapped_skills attribute
    assert all(doc._.mapped_skills for doc in job_ads_skills)
    # assert it has a skill_spans attribute
    assert all(doc._.skill_spans for doc in job_ads_skills)
    # assert that skills extracted is same len as job ad
    assert len(job_ads_skills) == len(job_ads)
