"""
Combine the labelling data from several folders
"""
import pandas as pd
from tqdm import tqdm

import re
from collections import defaultdict, Counter

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    get_s3_data_paths,
    load_s3_json,
    load_s3_data,
    save_to_s3,
)

from ojd_daps_skills import bucket_name

s3 = get_s3_resource()

# The labelling outputs and the metadata files relevant for their inputs
labelled_data_s3_folders = {
    "escoe_extension/outputs/skill_span_labels/": "escoe_extension/outputs/data/skill_ner/label_chunks/20220624_0_sample_labelling_metadata.json",
    "escoe_extension/outputs/labelled_job_adverts/LIZ_skill_spans/": "escoe_extension/outputs/data/skill_ner/label_chunks/20220819_3_sample_labelling_metadata.json",
    "escoe_extension/outputs/labelled_job_adverts/INDIA_skill_spans/": "escoe_extension/outputs/data/skill_ner/label_chunks/20220819_1_sample_labelling_metadata.json",
    "escoe_extension/outputs/labelled_job_adverts/CATH_skill_spans/": "escoe_extension/outputs/data/skill_ner/label_chunks/20220819_0_sample_labelling_metadata.json",
}


def load_original_metadata(labelled_data_s3_folders):
    metadata_jobids = {}
    for metadata_file in labelled_data_s3_folders.values():

        label_job_id_dict = load_s3_data(s3, bucket_name, metadata_file)
        label_job_id_dict = {int(k): v for k, v in label_job_id_dict.items()}
        metadata_jobids[metadata_file] = label_job_id_dict
    return metadata_jobids


def load_label_metadata(labelled_data_s3_folders, metadata_jobids):
    # Find the IDs of job adverts we want to include
    label_meta = []
    for folder_name in labelled_data_s3_folders.keys():
        meta_file_name = labelled_data_s3_folders[folder_name]
        meta_data = metadata_jobids[meta_file_name]

        file_names = get_s3_data_paths(s3, bucket_name, folder_name, "*")
        file_names.remove(folder_name)

        for file_name in tqdm(file_names):

            job_advert_labels = load_s3_json(s3, bucket_name, file_name)
            task_id = job_advert_labels["task"]["id"]
            label_id = job_advert_labels["task"].get("inner_id", task_id)
            label_meta.append(
                {
                    "id": job_advert_labels["id"],  # Unique ID for the labelling task
                    "label_id": label_id,  # ID of task index
                    "updated_at": job_advert_labels["updated_at"],
                    "was_cancelled": job_advert_labels["was_cancelled"],
                    "from_file_name": folder_name,
                    "job_id": meta_data.get(label_id - 1),
                }
            )
    label_meta = pd.DataFrame(label_meta)
    return label_meta


def filter_label_meta(label_meta):
    """
    Filter out the joined labelled metadata to get the IDs for non-duplicate job advert labels
    """
    sorted_df = label_meta.sort_values(by=["updated_at"], ascending=False)
    sorted_df = sorted_df[~sorted_df["was_cancelled"]]
    sorted_df.drop_duplicates(subset=["job_id"], keep="first", inplace=True)
    print(f"We will be using data from {len(sorted_df)} job adverts")
    keep_id_dict = (
        sorted_df.groupby("from_file_name")["id"].apply(lambda x: list(x)).to_dict()
    )

    return keep_id_dict


def combine_results(labelled_data_s3_folders, keep_id_dict, metadata_jobids):
    """
    Now you know what sample you are taking, merge the label results from the different files into one dict
    """
    job_labels = {}
    for folder_name in labelled_data_s3_folders.keys():
        # Get the label ids to keep for this folder of data
        keep_label_ids = keep_id_dict[folder_name]

        meta_file_name = labelled_data_s3_folders[folder_name]
        meta_data = metadata_jobids[meta_file_name]

        file_names = get_s3_data_paths(s3, bucket_name, folder_name, "*")
        file_names.remove(folder_name)

        for file_name in tqdm(file_names):
            job_advert_labels = load_s3_json(s3, bucket_name, file_name)
            task_id = job_advert_labels["task"]["id"]
            label_id = job_advert_labels["task"].get("inner_id", task_id)

            if job_advert_labels["id"] in keep_label_ids:
                if "ner" in job_advert_labels["task"]["data"].keys():
                    # For some of the labelled data the text is in the 'ner' key rather than 'text'
                    job_advert_labels["task"]["data"]["text"] = job_advert_labels[
                        "task"
                    ]["data"]["ner"]
                job_id = meta_data.get(label_id - 1)
                job_labels[job_id] = {
                    "text": job_advert_labels["task"]["data"]["text"],
                    "labels": job_advert_labels["result"],
                }

    return job_labels


if __name__ == "__main__":

    metadata_jobids = load_original_metadata(labelled_data_s3_folders)
    label_meta = load_label_metadata(labelled_data_s3_folders, metadata_jobids)

    keep_id_dict = filter_label_meta(label_meta)

    job_labels = combine_results(
        labelled_data_s3_folders, keep_id_dict, metadata_jobids
    )

    from datetime import datetime as date

    date_stamp = str(date.today().date()).replace("-", "")
    output_file = f"escoe_extension/outputs/labelled_job_adverts/combined_labels_{date_stamp}.json"
    save_to_s3(
        s3,
        bucket_name,
        job_labels,
        output_file,
    )
