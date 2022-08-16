import pandas as pd
import numpy as np


def load_toy_taxonomy():
    """
    A toy taxonomy for testing.

    Which ever taxonomy is loaded it should output the pandas dataframe taxonomy_skills
    and the dict of information/parameters in taxonomy_info
    """

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

    hier_name_mapper = {
        "K2": "computer",
        "K2.1": "computational skills",
        "S1": "communicate",
        "S1.1": "communicate with others",
        "S1.2": "communication skills",
    }

    taxonomy_info = {
        "num_hier_levels": num_hier_levels,
        "skill_type_dict": skill_type_dict,
        "match_thresholds_dict": match_thresholds_dict,
        "hier_name_mapper": hier_name_mapper,
        "skill_name_col": "description",
        "skill_id_col": "id",
        "skill_hier_info_col": "hierarchy_levels",
        "skill_type_col": "type",
    }

    return taxonomy_skills, taxonomy_info
