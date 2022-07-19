"""
ESCO- specific formating function to get ESCO data in the format needed for skill_ner_mapper.py


| id | description | type | hierarchy_levels |
|---|---|---|---|

id: A unique id for the skill/hierarchy
description: The skill/hierarchy level description text
type: What column name the skill/hier description is from (preferredLabel, altLabels, Level 2 preferred term, Level 3 preferred term)
hierarchy_levels: If a skill then which hierarchy levels is it in


"""

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)
from ojd_daps_skills import bucket_name

import re

import pandas as pd


def find_lev_0(code):
    return " ".join(re.findall("[a-zA-Z]+", code))


def split_up_code(code):
    """
    ESCO specific code splitting e.g. 'S4.8.1'-> ['S', S4', 'S4.8, 'S4.8.1']
    """
    lev_0 = find_lev_0(code)
    c = code.split(".")
    if len(c) == 1:
        return [lev_0, c[0], None, None]
    elif len(c) == 2:
        return [lev_0, c[0], ".".join(c[0:2]), None]
    else:
        return [lev_0, c[0], ".".join(c[0:2]), ".".join(c)]


def concepturi_2_tax(skills_concept_mapper, trans_skills_concept_mapper, concepturi):
    """
    Get the hierarchy codes for skills using the concept mappers.
    If the code is of length >10 then it wont be a hierarchy level code

    e.g. a skill concept code is '8f18f987-33e2-4228-9efb-65de25d03330' but a hierarchy code is 'S1.5.0'

    """
    codes = []
    # There may be multiple rows found for this uri, go through each one
    for uri in skills_concept_mapper[skills_concept_mapper["conceptUri"] == concepturi][
        "broaderUri"
    ].tolist():
        if "http://data.europa.eu/esco/skill/" in uri:
            code = uri.split("http://data.europa.eu/esco/skill/")[1]
            if len(code) < 10:
                codes.append(split_up_code(code))

    for uris in trans_skills_concept_mapper[
        trans_skills_concept_mapper["conceptUri"] == concepturi
    ]["broaderConceptUri"].tolist():
        for uri in uris.split(" | "):
            if "http://data.europa.eu/esco/skill/" in uri:
                code = uri.split("http://data.europa.eu/esco/skill/")[1]
                if len(code) < 10:
                    codes.append(split_up_code(code))

    return codes


if __name__ == "__main__":

    s3 = get_s3_resource()

    output_file_name = (
        "/escoe_extension/outputs/data/skill_ner_mapping/esco_data_formatted.csv"
    )

    # Load ESCO skills and hierarchy data

    skills_file_name = "escoe_extension/inputs/data/esco/skills_en.csv"
    hierarchy_file_name = "escoe_extension/inputs/data/esco/skillsHierarchy_en.csv"
    skill_file_name = "escoe_extension/inputs/data/esco/broaderRelationsSkillPillar.csv"
    transskill_file_name = (
        "escoe_extension/inputs/data/esco/transversalSkillsCollection_en.csv"
    )

    lev_2_name = "Level 2 preferred term"
    lev_3_name = "Level 3 preferred term"

    esco_skills = load_s3_data(s3, bucket_name, skills_file_name)
    esco_hierarchy = load_s3_data(s3, bucket_name, hierarchy_file_name)
    skills_concept_mapper = load_s3_data(s3, bucket_name, skill_file_name)
    skills_concept_mapper = skills_concept_mapper[
        skills_concept_mapper["broaderType"] == "SkillGroup"
    ]
    trans_skills_concept_mapper = load_s3_data(s3, bucket_name, transskill_file_name)

    # Get hierarchy codes for skills and clean
    esco_skills["hierarchy_levels"] = esco_skills["conceptUri"].apply(
        lambda x: concepturi_2_tax(
            skills_concept_mapper, trans_skills_concept_mapper, x
        )
    )
    esco_skills["id"] = esco_skills["conceptUri"].apply(lambda x: x.split("/")[-1])
    esco_skills["altLabels"] = esco_skills["altLabels"].apply(
        lambda x: x.split("\n") if isinstance(x, str) else x
    )

    # Separate out preferred and alternative labels in separate rows
    pref_label_skills = esco_skills[["id", "preferredLabel", "hierarchy_levels"]]
    pref_label_skills["type"] = ["preferredLabel"] * len(pref_label_skills)
    pref_label_skills.rename(columns={"preferredLabel": "description"}, inplace=True)

    alt_label_skills = esco_skills.explode("altLabels")[
        ["id", "altLabels", "hierarchy_levels"]
    ]
    alt_label_skills["type"] = ["altLabels"] * len(alt_label_skills)
    alt_label_skills.rename(columns={"altLabels": "description"}, inplace=True)

    # Get level 2 and 3 hierarchy information separately
    lev_2_skills = (
        esco_hierarchy[[lev_2_name, "Level 2 code"]].dropna().drop_duplicates()
    )
    lev_2_skills["type"] = ["level_2"] * len(lev_2_skills)
    lev_2_skills.rename(
        columns={lev_2_name: "description", "Level 2 code": "id"},
        inplace=True,
    )

    lev_3_skills = (
        esco_hierarchy[[lev_3_name, "Level 3 code"]].dropna().drop_duplicates()
    )
    lev_3_skills["type"] = ["level_3"] * len(lev_3_skills)
    lev_3_skills.rename(
        columns={lev_3_name: "description", "Level 3 code": "id"},
        inplace=True,
    )

    # Merge altogether and save
    esco_data = pd.concat(
        [pref_label_skills, alt_label_skills, lev_2_skills, lev_3_skills]
    )
    esco_data = esco_data[pd.notnull(esco_data["description"])].reset_index(drop=True)
    save_to_s3(s3, bucket_name, esco_data, output_file_name)
