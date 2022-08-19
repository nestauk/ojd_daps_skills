import pandas as pd
import numpy as np


def load_toy_taxonomy():
    """
    A toy taxonomy for testing.
    """

    taxonomy_skills = pd.DataFrame(
        {
            "type": [
                "skill",
                "skill",
                "skill_group_3",
                "skill_group_3",
                "skill_group_2",
            ],
            "description": [
                "use spreadsheets software",
                "use communication techniques",
                "communication, collaboration and creativity",
                "mathematics",
                "presenting information",
            ],
            "id": ["abcd", "cdef", "S1.0.0", "S1.2.1", "S1.4"],
            "hierarchy_levels": [
                [["S", "S5", "S5.6", "S5.6.1"], ["S", "S5", "S5.5", "S5.5.2"]],
                [["S", "S1", "S1.0", "S1.0.0"]],
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )
    taxonomy_skills["cleaned skills"] = taxonomy_skills["description"]

    return taxonomy_skills
