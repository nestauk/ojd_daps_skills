import pytest
import spacy
import itertools

from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills

es = ExtractSkills()

job_adverts = [
    "The job involves communication and maths skills",
    "The job involves excel and presenting skills. You need good excel skills",
]


def test_load_things():

    es.load_things()

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


def test_get_ner_skills():

    predicted_skills = es.get_ner_skills(job_adverts)

    assert isinstance(job_adverts, str) or isinstance(job_adverts, list)
    len(
        [
            i
            for i in predicted_skills
            if list(i.keys()) == ["EXPERIENCE", "SKILL", "MULTISKILL"]
        ]
    ) == len(predicted_skills)
    assert len(predicted_skills) == len(job_adverts)


def test_map_skills():

    predicted_skills = es.get_ner_skills(job_adverts)
    matched_skills = es.map_skills(predicted_skills)

    assert len(job_adverts) == len(matched_skills)
    assert len([i for i in matched_skills if isinstance(i, dict)]) == len(
        matched_skills
    )
    test_skills = list(
        itertools.chain(
            *[[skill[1][0] for skill in skills["SKILL"]] for skills in matched_skills]
        )
    )

    test_skills_not_in_taxonomy = []
    for test_skill in list(set(test_skills)):
        if test_skill not in list(es.taxonomy_skills["cleaned skills"]):
            test_skills_not_in_taxonomy.append(test_skill)

    assert (
        test_skills_not_in_taxonomy == []
    ), f"extracted '{test_skills_not_in_taxonomy}' skill not in taxonomy"


def test_extract_skills():

    extract_matched_skills = es.extract_skills(job_adverts)

    assert len(extract_matched_skills) == len(job_adverts)
    assert isinstance(extract_matched_skills, list)
