import pytest
import spacy
import itertools

from ojd_daps_skills.utils.text_cleaning import short_hash
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills

es = ExtractSkills()

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

def test_hardcoded_skills():
    job_adverts = ["you must be ambitious and resilient. You must also be able to use Excel"]
    extracted_matched_skills = es.extract_skills(job_adverts)
    #load hard coded matches
    hard_coded_skills = es.hard_coded_skills
    #
    hard_coded_excel = extracted_matched_skills[0]['SKILL'][0]
    assert hard_coded_excel[1][0] == hard_coded_skills.get(str(short_hash(hard_coded_excel[0])))['match_skill'] 
    #
    hard_coded_resilience = extracted_matched_skills[0]['SKILL'][1]
    assert hard_coded_resilience[1][0] == hard_coded_skills.get(str(short_hash(hard_coded_resilience[0])))['match_skill']


