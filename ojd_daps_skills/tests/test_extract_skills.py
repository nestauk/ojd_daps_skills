import pytest
import spacy
import itertools

from ojd_daps_skills.utils.text_cleaning import short_hash
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills

es = ExtractSkills(local=False)

job_adverts = [
    "The job involves communication and maths skills",
    "The job involves excel and presenting skills. You need good excel skills",
]


def test_load():

    es.load()

    assert isinstance(es.nlp, spacy.lang.en.English)
    assert es.labels == ("EXPERIENCE", "SKILL", "MULTISKILL")
    assert es.skill_mapper
    assert (
        len(
            set(es.taxonomy_skills.columns)
            - set(["id", "type", "description", "hierarchy_levels", "cleaned skills"])
        )
        == 0
    )


def test_get_skills():

    predicted_skills = es.get_skills(job_adverts)

    # The keys are the labels for every job prediction
    assert all(
        [
            len(set(p.keys()).intersection(set(es.labels))) == len(es.labels)
            for p in predicted_skills
        ]
    )
    assert isinstance(predicted_skills[0]["SKILL"], list)
    assert len(predicted_skills) == len(job_adverts)


def test_map_skills():

    predicted_skills = es.get_skills(job_adverts)
    matched_skills = es.map_skills(predicted_skills)

    assert len(job_adverts) == len(matched_skills)
    assert all([isinstance(i, dict) for i in matched_skills])
    test_skills = list(
        itertools.chain(
            *[[skill[1][0] for skill in skills["SKILL"]] for skills in matched_skills]
        )
    )
    assert (
        set(test_skills).difference(set(es.taxonomy_info["hier_name_mapper"].values()))
        == set()
    )


def test_map_no_skills():
    job_adverts = ["nothing", "we want excel skills", "we want communication skills"]
    extract_matched_skills = es.extract_skills(job_adverts)
    assert len(job_adverts) == len(extract_matched_skills)
