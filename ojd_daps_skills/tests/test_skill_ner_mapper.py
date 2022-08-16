import pytest

import pandas as pd
import numpy as np

from ojd_daps_skills.pipeline.skill_ner_mapping.skill_ner_mapper import SkillMapper

skill_mapper = SkillMapper(
    skill_name_col="description",
    skill_id_col="id",
    skill_hier_info_col="hierarchy_levels",
    skill_type_col="type",
)

num_hier_levels = 2
skill_type_dict = {
    "skill_types": ["preferredLabel", "altLabels"],
    "hier_types": ["level_2", "level_3"],
}

match_thresholds_dict = {
    "skill_match_thresh": 0.7,
    "top_tax_skills": {1: 0.5, 2: 0.5, 3: 0.5},
    "max_share": {1: 0, 2: 0.2, 3: 0.2},
}

taxonomy_skills = pd.DataFrame(
    {
        "type": [
            "preferredLabel",
            "preferredLabel",
            "altLabels",
            "level_2",
            "level_2",
            "level_3",
            "level_3",
        ],
        "description": [
            "microsoft excel",
            "communicate effectively",
            "communicate",
            "databases",
            "computational",
            "communications",
            "excel database",
        ],
        "id": ["p1", "p2", "a1", "l21", "l22", "l31", "l32"],
        "hierarchy_levels": [
            [["K2", "K2.1"]],
            [["S1", "S1.1"]],
            [["S1", "S1.2"]],
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
    }
)

taxonomy_skills["cleaned skills"] = taxonomy_skills["description"]

ojo_skills = {
    "predictions": {
        "a123": {"SKILL": ["communication skills", "microsoft excel"]},
        "b234": {"SKILL": ["excel skills", "communication skills"]},
        "c345": {"SKILL": ["filing"]},
    }
}


def test_preprocess_job_skills():
    clean_ojo_skills, skill_hashes = skill_mapper.preprocess_job_skills(ojo_skills)
    assert len(ojo_skills["predictions"]) == len(clean_ojo_skills)
    assert len(
        set([m for v in ojo_skills["predictions"].values() for m in v["SKILL"]])
    ) == len(skill_hashes)


def test_map_skills():

    clean_ojo_skills, skill_hashes = skill_mapper.preprocess_job_skills(ojo_skills)

    # ojo_esco_predefined = {-8387769020912651234: "p3"}
    # skill_hashes = skill_mapper.filter_skill_hash(skill_hashes, ojo_esco_predefined)

    skill_mapper.embed_taxonomy_skills(taxonomy_skills, "notneeded", save=False)
    skills_to_taxonomy = skill_mapper.map_skills(
        taxonomy_skills, skill_hashes, num_hier_levels, skill_type_dict
    )

    assert len(skills_to_taxonomy) == len(skill_hashes)

    hier_name_mapper = {
        "K2": "computer",
        "K2.1": "computational skills",
        "S1": "communicate",
        "S1.1": "communicate with others",
        "S1.2": "communication skills",
    }

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
        == "communicate with others"
    )

    # Check some basic features
    assert len(final_match) == len(skills_to_taxonomy)
    assert len(final_match) == len(skill_hashes)

    skill_hash_to_esco, final_ojo_skills = skill_mapper.link_skill_hash_to_job_id(
        clean_ojo_skills, final_match
    )

    assert final_ojo_skills.keys() == ojo_skills["predictions"].keys()
