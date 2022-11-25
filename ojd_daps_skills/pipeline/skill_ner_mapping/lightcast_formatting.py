"""
Lightcast- specific formating function to get lightcast data in the format needed for skill_ner_mapper.py

| id | description | type | hierarchy_levels |
|---|---|---|---|

id: A unique id for the skill/hierarchy
description: The skill/hierarchy level description text
type: What column name the skill/hier description is from (category, subcategory)
hierarchy_levels: If a skill then which hierarchy levels is it in

To run the script, python ojd_daps_skills/pipeline/skill_ner_mapping/lightcast_formatting.py --client-id CLIENT_ID --client-secret CLIENT_SECRET
"""

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    save_to_s3,
)
from ojd_daps_skills import bucket_name
from ojd_daps_skills.pipeline.evaluation.lightcast_evaluation import (
    get_lightcast_access_token,
)

import pandas as pd
from argparse import ArgumentParser
import requests
import numpy as np
import ast


def get_lightcast_skills(access_code: str) -> pd.DataFrame:
    """Call lightcast API to return Open Skills taxonomy.

    Inputs:
        access_code (str): Access code to query lightcast API

    Outputs:
        DataFrame of lightcast skills.
    """

    url = "https://emsiservices.com/skills/versions/latest/skills"

    querystring = {"fields": "id,name,type,category,subcategory"}

    headers = {"Authorization": "Bearer " + access_code}

    response = requests.request("GET", url, headers=headers, params=querystring)

    if response.ok:
        return pd.DataFrame(response.json()["data"])
    else:
        return response


def format_lightcast_skills(lightcast_skills: pd.DataFrame) -> pd.DataFrame:
    """Format lightcast skills taxonomy into format needed for
    skill_ner_mapper.py."""
    lightcast_skills[["category_id", "category_name"]] = pd.json_normalize(
        lightcast_skills["category"]
    )
    lightcast_skills[["subcategory_id", "subcategory_name"]] = pd.json_normalize(
        lightcast_skills["subcategory"]
    )

    lightcast_skills["hierarchy_levels"] = (
        lightcast_skills["category_id"].astype(str)
        + " "
        + lightcast_skills["subcategory_id"].astype(str)
    )
    lightcast_skills.hierarchy_levels = lightcast_skills.hierarchy_levels.str.split(" ")

    def add_columns(skills_df, level_type: str):
        skills_df["type"] = level_type

        if level_type == "skill":
            skills_df["hierarchy_levels"] = skills_df["hierarchy_levels"]
        else:
            skills_df["hierarchy_levels"] = np.nan

        return skills_df

    all_skills = lightcast_skills[["id", "name", "hierarchy_levels"]].rename(
        columns={"name": "description"}
    )
    all_skills = add_columns(all_skills, "skill")

    category_skills = (
        lightcast_skills[["category_id", "category_name"]]
        .drop_duplicates()
        .rename(columns={"category_id": "id", "category_name": "description"})
    )
    category_skills = add_columns(category_skills, "category")

    subcategory_skills = (
        lightcast_skills[["category_id", "subcategory_id", "subcategory_name"]]
        .drop_duplicates()
        .rename(columns={"subcategory_name": "description"})
    )
    subcategory_skills["id"] = (
        subcategory_skills["category_id"].astype(str)
        + "."
        + subcategory_skills["subcategory_id"].astype(str)
    )
    subcategory_skills["subcategory_id"] = subcategory_skills["subcategory_id"].astype(
        str
    )
    subcategory_skills_dict = subcategory_skills.set_index("subcategory_id")[
        "id"
    ].to_dict()
    subcategory_skills = subcategory_skills[["id", "description"]]
    subcategory_skills = add_columns(subcategory_skills, "subcategory")

    def map_subcategory_ids(hierarchy_levels):
        """map subcategory ids"""
        if isinstance(hierarchy_levels, list):
            mapped_subcategory_ids = []
            for level in hierarchy_levels:
                mapped_id = subcategory_skills_dict.get(level)
                if not mapped_id:
                    mapped_subcategory_ids.append(level)
                else:
                    mapped_subcategory_ids.append(mapped_id)

            return [mapped_subcategory_ids]
        else:
            return hierarchy_levels

    def remove_bad_hierarchy_levels(hierarchy_levels):
        bad_hierarchy_levels = [["0.0", "0.0.100.0"]]
        if hierarchy_levels == bad_hierarchy_levels:
            return np.nan
        else:
            return hierarchy_levels

    lightcast_formatted = pd.concat(
        [all_skills, category_skills, subcategory_skills]
    ).reset_index(drop=True)
    lightcast_formatted[
        "hierarchy_levels"
    ] = lightcast_formatted.hierarchy_levels.apply(map_subcategory_ids).apply(
        remove_bad_hierarchy_levels
    )
    lightcast_formatted = lightcast_formatted.query("description.notna()").query(
        'description != "NULL"'
    )

    return lightcast_formatted


if __name__ == "__main__":

    s3 = get_s3_resource()

    output_file_name = (
        "escoe_extension/outputs/data/skill_ner_mapping/lightcast_data_formatted.csv"
    )

    hier_name_mapper_file_name = (
        "escoe_extension/outputs/data/skill_ner_mapping/lightcast_hier_mapper.json"
    )

    parser = ArgumentParser()

    parser.add_argument(
        "--client-id", help="EMSI skills API client id",
    )

    parser.add_argument("--client-secret", help="EMSI skills API client secret.")

    args = parser.parse_args()

    client_id = args.client_id
    client_secret = args.client_secret

    access_code = get_lightcast_access_token(client_id, client_secret)
    lightcast_skills = get_lightcast_skills(access_code)
    lightcast_skills_formatted = format_lightcast_skills(lightcast_skills)

    hier_name_mapper = (
        lightcast_skills_formatted[lightcast_skills_formatted["type"] != "skill"][
            ["id", "description"]
        ]
        .dropna()
        .set_index("id")["description"]
        .to_dict()
    )
    save_to_s3(s3, bucket_name, hier_name_mapper, hier_name_mapper_file_name)
    save_to_s3(s3, bucket_name, lightcast_skills_formatted, output_file_name)
