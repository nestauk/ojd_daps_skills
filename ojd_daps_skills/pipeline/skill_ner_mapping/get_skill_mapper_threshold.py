"""
Defines skill mapper threshold based on labelling
skill matches between threshold windows.

python -thresh 0.7 get_skill_mapper_threshold.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser

from ojd_daps_skills import bucket_name, config
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    get_s3_data_paths,
)
from ojd_daps_skills.pipeline.skill_ner_mapping.skill_ner_mapper_utils import get_top_skill_score_df

def evaluate_skill_matches(labelled_df: pd.DataFrame) -> dict:
    """Calculates the true and false positives and negatives per threshold for
    the labelled skill skill matches.

    Inputs:
        skill_span_labels (pd.DataFrame): DataFrame of labelled skill matches.

    Outputs:
        label_results (dict): Dictionary of confusion matrix per threshold level.
    """
    threshold_results = dict()
    for threshold_window, labelled_data in labelled_df.groupby("threshold_window"):
        labelled_data_dedup = labelled_data.drop_duplicates("ojo_ner_skills")
        y_true = [1 for _ in range(0, len(labelled_data_dedup["label"]))]
        y_pred = list(labelled_data_dedup["label"])
        threshold_results[threshold_window] = accuracy_score(y_pred, y_true)

    return threshold_results


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--thresh", help="similarity threshold number", default=0.7,
    )

    args = parser.parse_args()
    threshold = args.thresh

    # load data
    labelled_dir = get_s3_data_paths(
        get_s3_resource(),
        bucket_name,
        "escoe_extension/outputs/evaluation/skill_mappings/",
        "*.csv",
    )

    labelled_dfs = []
    for labelled_match in labelled_dir:
        labelled_dfs.append(
            load_s3_data(get_s3_resource(), bucket_name, labelled_match)
        )

    ojo_to_esco = load_s3_data(
        get_s3_resource(), bucket_name, config["skills_ner_mapping_esco"]
    )

    ojo_to_esco_df = get_top_skill_score_df(ojo_to_esco, 'esco')

    # evaluate labelled skill matches
    for labelled_df in labelled_dfs:
        print(evaluate_skill_matches(labelled_df))

    # based off of scores with high true positives
    print(
        f"if the threshold is {float(threshold)}, we will label {len(ojo_to_esco_df[ojo_to_esco_df.top_scores > float(threshold)])/len(ojo_to_esco_df)} percent of skills."
    )
