"""
Test cases for the SkillsMapper class.
"""

import pytest

from ojd_daps_skills.extract_skills.extract_skills_utils import ExtractConfig
from ojd_daps_skills.map_skills.skill_mapper_utils import MapConfig
from ojd_daps_skills.map_skills.skill_mapper import SkillsMapper


@pytest.fixture
def skill_mapper():
    map_config = MapConfig.create(taxonomy_name="esco")
    sm = SkillsMapper(config=map_config)
    return sm


@pytest.fixture
def extract_config():
    return ExtractConfig.create("nestauk/en_skillner", "nestauk/multiskill-classifier")


@pytest.fixture
def skill_embeddings(skill_mapper):
    skills = ["Python", "Oral Communication"]
    skill_embeddings = skill_mapper.config.bert_model.transform(skills)
    return skill_embeddings


@pytest.fixture
def taxonomy_embeddings_dict(skill_mapper):
    return skill_mapper.config.taxonomy_embeddings


@pytest.fixture
def job_ads(extract_config):
    job_texts = [
        "We are hiring for a data scientist with experience in Python and machine learning.",
        "We are looking for a data analyst with experience in SQL and oral communication.",
    ]

    job_docs = [extract_config.nlp(job_text) for job_text in job_texts]
    for job_doc in job_docs:
        # kind of arbrary but we need to set skill spans for the SkillsMapper
        job_doc._.skill_spans = [
            "Python",
            "machine learning",
            "SQL",
            "oral communication",
        ]

    return job_docs


# def test_get_top_taxonomy_skills(
#     skill_mapper, skill_embeddings, taxonomy_embeddings_dict
# ):
#     top_taxonomy_skills = skill_mapper.get_top_taxonomy_skills(
#         skill_embeddings, taxonomy_embeddings_dict
#     )
#     # assert that its a tuple
#     assert isinstance(top_taxonomy_skills, tuple)
#     assert len(top_taxonomy_skills) == 3
#     assert isinstance(top_taxonomy_skills[0], list)
#     assert isinstance(top_taxonomy_skills[1], list)
#     assert type(top_taxonomy_skills[2]) is pd.core.indexes.base.Index
#     assert len(top_taxonomy_skills[1][0]) == 2


# def test_get_top_hierarchy_skills(
#     skill_mapper, skill_embeddings, taxonomy_embeddings_dict
# ):
#     top_hierarchy_skills = skill_mapper.get_top_hierarchy_skills(
#         skill_embeddings, taxonomy_embeddings_dict
#     )

#     assert isinstance(top_hierarchy_skills, tuple)
#     assert len(top_hierarchy_skills) == 2
#     assert isinstance(top_hierarchy_skills[0], dict)
#     assert isinstance(top_hierarchy_skills[1], dict)
#     assert ["top_sim_indxs", "top_sim_scores", "taxonomy_skills_ix"] == list(
#         top_hierarchy_skills[0][0].keys()
#     )
#     assert top_hierarchy_skills[1] == {0: "skill_group_2", 1: "skill_group_3"}


# def test_get_embeddings(skill_mapper, job_ads):
#     job_embeddings = skill_mapper.get_embeddings(job_ads)

#     assert isinstance(job_embeddings, tuple)
#     assert len(job_embeddings) == 2
#     assert isinstance(job_embeddings[0], np.ndarray)
#     assert isinstance(job_embeddings[1], dict)

#     assert job_embeddings[0].shape[0] == 4
#     assert len(job_embeddings[1]) == 5


# def test_map_skills(skill_mapper, job_ads):
#     mapped_skills = skill_mapper.map_skills(job_ads)

#     assert isinstance(mapped_skills, list)
#     assert len(mapped_skills) == 4
#     assert all(isinstance(skill, dict) for skill in mapped_skills)


# def test_match_skills(skill_mapper, job_ads):
#     matched_skills = skill_mapper.match_skills(job_ads)

#     assert isinstance(matched_skills, dict)
#     assert len(matched_skills) == 4
#     # assert that all the keys are ints
#     assert all(isinstance(key, int) for key in matched_skills.keys())
#     # assert that all the values are dictioanries
#     assert all(isinstance(value, dict) for value in matched_skills.values())
