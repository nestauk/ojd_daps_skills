"""
Defines skill mapper threshold based on labelling
skill matches between threshold windows.  
"""
########################################################################
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from ojd_daps_skills import bucket_name, config
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    get_s3_data_paths,
)

########################################################################


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
        results = confusion_matrix(y_true, y_pred).ravel()
        if results.shape[0] == 4:
            threshold_results[threshold_window] = {
                "false_negative": results[2] / len(labelled_data_dedup),
                "true_positive": results[3] / len(labelled_data_dedup),
            }
        else:
            threshold_results[threshold_window] = {
                "false_negative": 0,
                "true_positive": results[0] / len(labelled_data_dedup),
            }

    return threshold_results


if __name__ == "__main__":

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

    ojo_to_esco_df = pd.DataFrame(ojo_to_esco).T
    for col in ("esco_taxonomy_skills", "esco_taxonomy_scores"):
        col_name = "top_" + col.split("_")[-1]
        ojo_to_esco_df[col_name] = ojo_to_esco_df[col].apply(
            lambda x: [i[0] for i in x]
        )

    ojo_to_esco_df = ojo_to_esco_df.apply(pd.Series.explode)[
        ["ojo_ner_skills", "top_skills", "top_scores"]
    ]

    # evaluate labelled skill matches
    for labelled_df in labelled_dfs:
        print(evaluate_skill_matches(labelled_df))

    # based off of scores with high true positives
    print(
        f"if the threshold is 0.72, we will label {len(ojo_to_esco_df[ojo_to_esco_df.top_scores > 0.72])/len(ojo_to_esco_df)} percent of skills."
    )
    print(
        f"if the threshold is 0.73, we will label {len(ojo_to_esco_df[ojo_to_esco_df.top_scores > 0.73])/len(ojo_to_esco_df)} percent of skills."
    )
    print(
        f"if the threshold is 0.74, we will label {len(ojo_to_esco_df[ojo_to_esco_df.top_scores > 0.74])/len(ojo_to_esco_df)} percent of skills."
    )
