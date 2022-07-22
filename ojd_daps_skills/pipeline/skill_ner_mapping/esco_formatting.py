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
from collections import defaultdict

import pandas as pd


def find_lev_0(code):
    return " ".join(re.findall("[a-zA-Z]+", code))


def split_up_code(code):
    """
    ESCO specific code splitting e.g. 'S4.8.1'-> ['S', S4', 'S4.8, 'S4.8.1']
    """
    lev_0 = find_lev_0(code)
    c = code.split(".")
    if len(code) == 1:  # e.g. just "S"
        return [code, None, None, None]
    elif len(c) == 1:  # e.g. ["S1"]
        return [lev_0, c[0], None, None]
    elif len(c) == 2:
        return [lev_0, c[0], ".".join(c[0:2]), None]
    else:
        return [lev_0, c[0], ".".join(c[0:2]), ".".join(c)]


def split_up_isced_code(code):
    """
    isced codes are the knowledge codes, they are in the form "0901"
    The first 2 digits are highest level.
    Problems will arise if ESCO introduce more than 9 groups per level :/
    e.g. '0000' -> ['K', 'K00', 'K000', 'K0000']
    We introduce a 'K' to make the knowledge link more obvious.
    """
    if code.isnumeric():
        c = "K" + code
        if len(c) == 2:
            return [c[0], c[0:3], None, None]
        elif len(c) == 3:
            return [c[0], c[0:3], c[0:4], None]
        else:
            return [c[0], c[0:3], c[0:4], c]
    else:
        return split_up_code(code)


def concepturi_2_tax(skills_concept_mapper, trans_skills_concept_mapper):
    """
    Create a concept ID to hierarchy codes mapper dict.
    If the code is of length >10 then it wont be a hierarchy level code

    e.g. a skill concept code is '8f18f987-33e2-4228-9efb-65de25d03330' but a hierarchy code is 'S1.5.0'

    """
    # 2 step process,
    # 1. find mappings from concept id to broader id (as long as this is len <10 - otherwise its actually a concept id)
    # 2. go through broader ids which were >=10 (i.e. concept ids) and try to find them in the step 1 mapping dict, update
    # Add the transversal skills to this too in the same way

    concept_mapping_df_concat = pd.concat(
        [skills_concept_mapper, trans_skills_concept_mapper]
    )

    concept_mapper = defaultdict(list)
    for i, row in concept_mapping_df_concat.iterrows():
        uri = row["conceptUri"]
        concept_code = uri.split("/")[-1]
        if "isced" not in uri:
            # Skill mappings
            # Only transversal data uses 'broaderConceptUri' column, otherwise 'broaderUri' column
            broader_uris = (
                row["broaderConceptUri"]
                if pd.notnull(row["broaderConceptUri"])
                else row["broaderUri"]
            )  # can be multiple
            for broader_uri in broader_uris.split(" | "):
                broader_code = broader_uri.split("/")[-1]
                if "isced" not in broader_uri:
                    if len(broader_code) < 10:
                        concept_mapper[concept_code].append(split_up_code(broader_code))
                else:
                    concept_mapper[concept_code].append(
                        split_up_isced_code(broader_code)
                    )
        else:
            # Map the isced codes (knowledge)
            # These just require cleaning up the concept code
            concept_mapper[concept_code].append(split_up_isced_code(concept_code))

    not_found = []
    for i, row in concept_mapping_df_concat.iterrows():
        uri = row["conceptUri"]
        concept_code = uri.split("/")[-1]
        broader_uris = (
            row["broaderConceptUri"]
            if pd.notnull(row["broaderConceptUri"])
            else row["broaderUri"]
        )  # can be multiple
        for broader_uri in broader_uris.split(" | "):
            broader_code = broader_uri.split("/")[-1]
            if len(broader_code) >= 10:
                if concept_mapper.get(broader_code):
                    concept_mapper[concept_code].extend(
                        concept_mapper.get(broader_code)
                    )
                else:
                    not_found.append(broader_code)
    # Get rid of duplicates e.g. [['K', 'K00', 'K00', None], ['K', 'K00', 'K00', None], ['S', 'S2', None, None]]
    concept_mapper = {
        k: [list(item) for item in set(tuple(row) for row in v)]
        for k, v in concept_mapper.items()
        if k != []
    }

    return concept_mapper


def get_esco_hier_mapper(esco_hierarchy, knowledge_skills):
    """
    Create a dictionary from esco hierarchy codes to level name
    """

    esco_mapper = {
        k: v
        for k, v in esco_hierarchy.set_index("Level 1 code")
        .to_dict()["Level 1 preferred term"]
        .items()
        if pd.notnull(k)
    }
    esco_mapper.update(
        {
            k: v
            for k, v in esco_hierarchy.set_index("Level 2 code")
            .to_dict()["Level 2 preferred term"]
            .items()
            if pd.notnull(k)
        }
    )
    esco_mapper.update(
        {
            k: v
            for k, v in esco_hierarchy.set_index("Level 3 code")
            .to_dict()["Level 3 preferred term"]
            .items()
            if pd.notnull(k)
        }
    )

    def get_isco_name(concepturi):
        concept_code = concepturi.split("/")[-1]
        if ("isced" in concepturi) and (concept_code.isnumeric()):
            c = "K" + concept_code
            return c
        else:
            return None

    knowledge_skills["conceptUri_mapped"] = knowledge_skills["conceptUri"].apply(
        lambda x: get_isco_name(x)
    )

    ## Add to level mappers
    for k, v in (
        knowledge_skills.set_index("conceptUri_mapped")
        .to_dict()["preferredLabel"]
        .items()
    ):
        if pd.notnull(k):
            if len(k) == 3:
                esco_mapper[k] = v
            elif len(k) == 4:
                esco_mapper[k] = v
            elif len(k) == 5:
                esco_mapper[k] = v
    return esco_mapper


if __name__ == "__main__":

    s3 = get_s3_resource()

    output_file_dir = "escoe_extension/outputs/data/skill_ner_mapping/"

    output_mapper_file_name = output_file_dir + "esco_hier_mapper.json"
    output_file_name = output_file_dir + "esco_data_formatted.csv"

    # Load ESCO skills and hierarchy data

    skills_file_name = "escoe_extension/inputs/data/esco/skills_en.csv"
    hierarchy_file_name = "escoe_extension/inputs/data/esco/skillsHierarchy_en.csv"
    skill_file_name = "escoe_extension/inputs/data/esco/broaderRelationsSkillPillar.csv"
    transskill_file_name = (
        "escoe_extension/inputs/data/esco/transversalSkillsCollection_en.csv"
    )
    skill_groups_file_name = "escoe_extension/inputs/data/esco/skillGroups_en.csv"

    lev_2_name = "Level 2 preferred term"
    lev_3_name = "Level 3 preferred term"

    esco_skills = load_s3_data(s3, bucket_name, skills_file_name)
    esco_hierarchy = load_s3_data(s3, bucket_name, hierarchy_file_name)
    skills_concept_mapper = load_s3_data(s3, bucket_name, skill_file_name)
    trans_skills_concept_mapper = load_s3_data(s3, bucket_name, transskill_file_name)
    knowledge_skills = load_s3_data(s3, bucket_name, skill_groups_file_name)

    # Create a name mapper with the data which will be useful in later pipeline steps
    esco_mapper = get_esco_hier_mapper(esco_hierarchy, knowledge_skills)
    save_to_s3(s3, bucket_name, esco_mapper, output_mapper_file_name)

    # Concatenate the skills and the knowledge skills
    esco_skills = pd.concat([esco_skills, knowledge_skills])

    # Get hierarchy codes for skills and clean
    concept_mapper = concepturi_2_tax(
        skills_concept_mapper, trans_skills_concept_mapper
    )
    esco_skills["id"] = esco_skills["conceptUri"].apply(lambda x: x.split("/")[-1])
    esco_skills["hierarchy_levels"] = esco_skills["id"].map(concept_mapper)
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

    print(
        f"Removing {sum(pd.isnull(pref_label_skills['hierarchy_levels']))} out of {len(pref_label_skills)} preferred label skills weren't mapped"
    )
    print(
        f"Removing {sum(pd.isnull(alt_label_skills['hierarchy_levels']))} out of {len(alt_label_skills)} alternative label skills weren't mapped"
    )

    not_found_concept_id = list(
        set(
            pref_label_skills[pd.isnull(pref_label_skills["hierarchy_levels"])][
                "id"
            ].tolist()
            + alt_label_skills[pd.isnull(alt_label_skills["hierarchy_levels"])][
                "id"
            ].tolist()
        )
    )

    # Remove data not mapped
    pref_label_skills = pref_label_skills[
        pd.notnull(pref_label_skills["hierarchy_levels"])
    ]
    alt_label_skills = alt_label_skills[
        pd.notnull(alt_label_skills["hierarchy_levels"])
    ]
    print(f"{len(pref_label_skills)} remaining preferred labels")
    print(f"{len(alt_label_skills)} remaining alternate labels")

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
