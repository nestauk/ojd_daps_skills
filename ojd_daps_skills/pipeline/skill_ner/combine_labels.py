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
    load_prodigy_jsonl_s3_data,
)

from ojd_daps_skills import bucket_name

s3 = get_s3_resource()

# The Label-Studio labelling outputs and the metadata files relevant for their inputs
labelled_data_s3_folders = {
    "escoe_extension/outputs/skill_span_labels/": "escoe_extension/outputs/data/skill_ner/label_chunks/20220624_0_sample_labelling_metadata.json",
    "escoe_extension/outputs/labelled_job_adverts/LIZ_skill_spans/": "escoe_extension/outputs/data/skill_ner/label_chunks/20220819_3_sample_labelling_metadata.json",
    "escoe_extension/outputs/labelled_job_adverts/INDIA_skill_spans/": "escoe_extension/outputs/data/skill_ner/label_chunks/20220819_1_sample_labelling_metadata.json",
    "escoe_extension/outputs/labelled_job_adverts/CATH_skill_spans/": "escoe_extension/outputs/data/skill_ner/label_chunks/20220819_0_sample_labelling_metadata.json",
}

# The Prodigy labelled data
prodigy_labelled_data_s3_folder = "escoe_extension/outputs/labelled_job_adverts/prodigy/labelled_dataset_skills_080823.jsonl"


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
                    "type": "label-studio",
                }

    return job_labels


def load_format_prodigy(prodigy_labelled_data_s3_folder):
    """
    Load all prodigy labels
    Since these were labelled in 5 sentence chunks, then sort them into a nested dict
    with the job advert id and the sentence chunk number

    """
    s3 = get_s3_resource()
    prodigy_data_chunks = defaultdict(dict)
    prodigy_data = load_prodigy_jsonl_s3_data(
        s3, bucket_name, prodigy_labelled_data_s3_folder
    )
    for p in prodigy_data:
        if p["answer"] == "accept":
            prodigy_data_chunks[str(p["meta"]["id"])][p["meta"]["chunk"]] = p
    return prodigy_data_chunks


def combine_prodigy_spans(prodigy_data_chunks):
    """
    Since the prodigy data was labelled in 5 sentence chunks, we need
    to merge all of these chunks per advert including updating the span start and end
    characters to fit with merged text
    """

    not_equal_spans_count = 0
    prodigy_job_labels = {}
    for job_id, job_adv_labels in prodigy_data_chunks.items():
        # Make sure the sentence chunks are in the correct order
        job_adv_labels = {k: job_adv_labels[k] for k in sorted(job_adv_labels)}

        # Combine texts and spans for each job advert
        full_text = []
        all_labels = []
        total_chars = 0
        for chunk_labels in job_adv_labels.values():
            full_text.append(chunk_labels["text"])
            for spans_info in chunk_labels["spans"]:
                all_labels.append(
                    {
                        "value": {
                            "start": spans_info["start"] + total_chars,
                            "end": spans_info["end"] + total_chars,
                            "text": chunk_labels["text"][
                                spans_info["start"] : spans_info["end"]
                            ],
                            "labels": [spans_info["label"]],
                        },
                        "id": (chunk_labels["_input_hash"], chunk_labels["_task_hash"]),
                        "origin": chunk_labels["_annotator_id"],
                    }
                )
            total_chars += (
                len(chunk_labels["text"]) + 2
            )  # plus two since we combine the 5 sentence chunks together with ". " at the end

        full_text = ". ".join(full_text)

        # checks
        for v in all_labels:
            if v["value"]["text"] != full_text[v["value"]["start"] : v["value"]["end"]]:
                not_equal_spans_count += 1
        if not_equal_spans_count != 0:
            print(
                f"There were {not_equal_spans_count} issues with merging these spans. Please investigate"
            )

        # Final output
        prodigy_job_labels[job_id] = {
            "text": full_text,
            "labels": all_labels,
            "type": "prodigy",
        }

    return prodigy_job_labels


if __name__ == "__main__":

    metadata_jobids = load_original_metadata(labelled_data_s3_folders)
    label_meta = load_label_metadata(labelled_data_s3_folders, metadata_jobids)

    keep_id_dict = filter_label_meta(label_meta)

    job_labels = combine_results(
        labelled_data_s3_folders, keep_id_dict, metadata_jobids
    )

    prodigy_data = load_format_prodigy(prodigy_labelled_data_s3_folder)
    prodigy_job_labels = combine_prodigy_spans(prodigy_data)

    # Merge label-studio and prodigy labels
    job_labels.update(prodigy_job_labels)
    print(f"We will be using data from {len(job_labels)} job adverts")

    from datetime import datetime as date

    date_stamp = str(date.today().date()).replace("-", "")
    output_file = f"escoe_extension/outputs/labelled_job_adverts/combined_labels_{date_stamp}.json"
    save_to_s3(
        s3,
        bucket_name,
        job_labels,
        output_file,
    )
