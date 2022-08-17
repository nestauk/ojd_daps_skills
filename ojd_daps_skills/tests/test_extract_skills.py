import pytest
from ojd_daps_skills.analysis.esco_datasets.
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills
from ojd_daps_skills.pipeline.extract_skills.extract_skills_utils import (
    load_toy_taxonomy,
)

extract_skills = ExtractSkills()

job_adverts = [
    "The job involves communication and maths skills",
    "The job involves excel and presenting skills. You need good excel skills",
]

def test_load_things():

    assert isinstance(extract_skills.nlp, spacy.lang.en.English)
    assert extract_skills.labels == ('EXPERIENCE', 'SKILL', 'MULTISKILL')
    assert (extract_skills.skill_mapper)
    assert len(set(extract_skills.taxonomy_skills.columns) - set(['id', 'type', 'description', 'hierarchy_levels', 'cleaned skills'])) == 0


def test_get_ner_skills():

    predicted_skills = extract_skills.get_ner_skills(job_adverts)
    len([i for i in predicted_skills if list(i.keys()) == ['EXPERIENCE', 'SKILL', 'MULTISKILL']]) == len(predicted_skills)
    #

def test_map_skills():

def test_extract_skills():

