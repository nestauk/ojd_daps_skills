import pytest
import spacy
import itertools

from ojd_daps_skills.utils.text_cleaning import short_hash
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills

es = ExtractSkills(local=True)

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


def test_hardcoded_mapping():
    """
    The mapped results using the algorithm should be the same as the hardcoded results
    """

    hard_coded_skills = {
        "3267542715426065": {
            "ojo_skill": "caring",
            "match_skill": "assisting and caring",
            "match_id": "S3.0.0",
        }
    }

    # The toy taxonomy doesn't have a hard coded skill input
    es.hard_coded_skills = hard_coded_skills
    hardcoded_matches = {
        h["ojo_skill"]: (h["match_skill"], h["match_id"])
        for h in hard_coded_skills.values()
    }

    mapped_skills = es.map_skills(
        [
            {"SKILL": [skill], "MULTISKILL": [], "EXPERIENCE": []}
            for skill in hardcoded_matches.keys()
        ]
    )

    assert type(mapped_skills) == list
    assert len(mapped_skills) == len(hard_coded_skills)

    correct_matches = []
    for mapped_skill in mapped_skills:
        skill = mapped_skill["SKILL"][0][0]
        mapped_result = mapped_skill["SKILL"][0][1]
        hardcoded_result = hardcoded_matches[skill]
        correct_matches.append(mapped_result == hardcoded_result)

    assert all(correct_matches)

    first_skill = mapped_skills[0]["SKILL"][0][0]
    assert type(hardcoded_matches[first_skill][0]) == str
