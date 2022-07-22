"""
"Generate random sample of skill span matches
per threshold score.

To run script,

python get_skill_mapper_threshold_sample.py --min 0.3 --max 1 --threshold_len 10 --sample_size 20
"""

import numpy as np
import pandas as pd
from argparse import ArgumentParser

from ojd_daps_skills import config, bucket_name
from ojd_daps_skills.pipeline.skill_ner_mapping.skill_ner_mapper_utils import get_top_skill_score_df
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)

# %%
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--min", help="minimum cosine similarity threshold", default=0.3,
    )

    parser.add_argument(
        "--max", help="maximum cosine similarity threshold", default=1.0,
    )

    parser.add_argument(
        "--threshold_len", help="number of thresholds", default=10,
    )

    parser.add_argument(
        "--sample_size", help="number of skill matches to label", default=30,
    )

    args = parser.parse_args()

    min_threshold = float(args.min)
    max_threshold = float(args.max)
    threshold_len = int(args.threshold_len)
    sample_size = args.sample_size

    # load data
    skills_to_esco = load_s3_data(
        get_s3_resource(), bucket_name, config["skills_ner_mapping_esco"]
    )
    skills_to_esco_df = get_top_skill_score_df(skills_to_esco, 'esco')
    # generate threshold list
    thresholds = [
        round(_, 2)
        for _ in list(np.linspace(min_threshold, max_threshold, threshold_len))
    ]
    # generate samples
    skill_spans_to_label = []
    for i in range(1, len(thresholds)):
        skill_threshs = skills_to_esco_df[
            skills_to_esco_df.top_scores.astype(float).between(
                thresholds[i - 1], thresholds[i]
            )
        ]
        skill_threshs_sample = skill_threshs.sample(
            int(sample_size), replace=True, random_state=42
        )
        skill_threshs_sample["threshold_window"] = (
            str(thresholds[i - 1]) + " - " + str(thresholds[i])
        )
        skill_spans_to_label.append(skill_threshs_sample)

    save_to_s3(
        get_s3_resource(),
        bucket_name,
        pd.concat(skill_spans_to_label).reset_index(drop=True),
        f"/escoe_extension/inputs/data/skill_mappings/skill_mappings_to_label_{min_threshold}_{max_threshold}_{threshold_len}.csv",
    )
