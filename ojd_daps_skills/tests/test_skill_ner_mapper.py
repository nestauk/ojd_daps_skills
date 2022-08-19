import pytest
import yaml
import os

import pandas as pd
import numpy as np

from ojd_daps_skills.pipeline.skill_ner_mapping.skill_ner_mapper import SkillMapper
from ojd_daps_skills import PROJECT_DIR
from ojd_daps_skills.pipeline.extract_skills.extract_skills_utils import (
    load_toy_taxonomy,
)

config_path = os.path.join(
    PROJECT_DIR, "ojd_daps_skills/config/extract_skills_toy.yaml"
)

with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

skill_mapper = SkillMapper(
    skill_name_col=config["skill_name_col"],
    skill_id_col=config["skill_id_col"],
    skill_hier_info_col=config["skill_hier_info_col"],
    skill_type_col=config["skill_type_col"],
)

taxonomy_skills = load_toy_taxonomy()

num_hier_levels = config["num_hier_levels"]
skill_type_dict = config["skill_type_dict"]
hier_name_mapper = config["hier_name_mapper"]

match_thresholds_dict = config["match_thresholds_dict"]

taxonomy_skills["cleaned skills"] = taxonomy_skills["description"]

ojo_skills = {
    "predictions": {
        "a123": {
            "SKILL": ["communication skills", "microsoft excel"],
            "MULTISKILL": [],
            "EXPERIENCE": [],
        },
        "b234": {
            "SKILL": ["excel skills", "communication skills"],
            "MULTISKILL": ["verbal and presentation skills"],
            "EXPERIENCE": [],
        },
        "c345": {"SKILL": ["filing"], "MULTISKILL": [], "EXPERIENCE": []},
    }
}


def test_preprocess_job_skills():
    clean_ojo_skills, skill_hashes = skill_mapper.preprocess_job_skills(ojo_skills)
    assert len(ojo_skills["predictions"]) == len(clean_ojo_skills)
    assert len(
        set(
            [
                m
                for v in ojo_skills["predictions"].values()
                for m in v["SKILL"] + v["MULTISKILL"]
            ]
        )
    ) == len(skill_hashes)


def test_map_skills():

    clean_ojo_skills, skill_hashes = skill_mapper.preprocess_job_skills(ojo_skills)

    # ojo_esco_predefined = {-8387769020912651234: "p3"}
    # skill_hashes = skill_mapper.filter_skill_hash(skill_hashes, ojo_esco_predefined)

    skill_mapper.embed_taxonomy_skills(taxonomy_skills)
    skills_to_taxonomy = skill_mapper.map_skills(
        taxonomy_skills, skill_hashes, num_hier_levels, skill_type_dict
    )

    assert len(skills_to_taxonomy) == len(skill_hashes)

    final_match = skill_mapper.final_prediction(
        skills_to_taxonomy,
        hier_name_mapper,
        match_thresholds_dict,
        num_hier_levels,
    )

    final_match_dict = {f["ojo_job_skill_hash"]: f for f in final_match}

    skill_hashes_rev = {v: k for k, v in skill_hashes.items()}

    # Check output links correctly
    ojo_skill_text = "communication skills"
    assert (
        final_match_dict[skill_hashes_rev[ojo_skill_text]]["ojo_skill"]
        == ojo_skill_text
    )

    # Check match is correct
    assert (
        final_match_dict[skill_hashes_rev[ojo_skill_text]]["match_skill"]
        == "communication, collaboration and creativity"
    )

    # Check some basic features
    assert len(final_match) == len(skills_to_taxonomy)
    assert len(final_match) == len(skill_hashes)

    skill_hash_to_esco, final_ojo_skills = skill_mapper.link_skill_hash_to_job_id(
        clean_ojo_skills, final_match
    )

    assert final_ojo_skills.keys() == ojo_skills["predictions"].keys()
