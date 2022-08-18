import pandas as pd
import numpy as np


def load_toy_taxonomy():
    """
    A toy taxonomy for testing.
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

    return taxonomy_skills
