"""
This script processes a sample of job adverts into a format needed for labelling
using label-studio.

Label-studio inputs a txt file where each row is a labelling task. Thus we output two files:
1. A txt file where each line is a job advert.
2. A json of line ID: job advert ID, so we keep track of what job advert each line of text was from.

"""

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)

import spacy
from tqdm import tqdm

from datetime import datetime as date
import json
import os

BUCKET_NAME = "open-jobs-lake"
S3_FOLDER = "escoe_extension/inputs/data/skill_ner/"
JOBS_SAMPLE = "data_sample/20220622_sampled_job_ads.json"
S3_OUTPUT_FOLDER = "escoe_extension/outputs/data/skill_ner/label_chunks/"

if __name__ == "__main__":

    s3 = get_s3_resource()

    file_name = os.path.join(S3_FOLDER, JOBS_SAMPLE)
    print(f"Processing job advert sample from {file_name}")

    sample_data = load_s3_data(s3, BUCKET_NAME, file_name)
    date_stamp = str(date.today().date()).replace("-", "")

    output_id = 0
    line_index = 0
    texts = []
    index_metadata = {}
    for job_id, job_info in sample_data.items():
        # If there are any description texts with '\n' in this will
        # mess the sentence separation up in the output step,
        # so just make sure they are all removed.
        texts.append(job_info["description"].replace("\n", " "))
        index_metadata[line_index] = job_id
        line_index += 1
        if line_index == 400:
            # Output to S3
            save_to_s3(
                s3,
                BUCKET_NAME,
                "\n".join(texts),
                os.path.join(
                    S3_OUTPUT_FOLDER,
                    f"{date_stamp}_{output_id}_sample_labelling_text_data.txt",
                ),
            )
            save_to_s3(
                s3,
                BUCKET_NAME,
                index_metadata,
                os.path.join(
                    S3_OUTPUT_FOLDER,
                    f"{date_stamp}_{output_id}_sample_labelling_metadata.json",
                ),
            )
            output_id += 1
            line_index = 0
            texts = []
            index_metadata = {}
    # Output last bit to S3
    save_to_s3(
        s3,
        BUCKET_NAME,
        "\n".join(texts),
        os.path.join(
            S3_OUTPUT_FOLDER, f"{date_stamp}_{output_id}_sample_labelling_text_data.txt"
        ),
    )
    save_to_s3(
        s3,
        BUCKET_NAME,
        index_metadata,
        os.path.join(
            S3_OUTPUT_FOLDER, f"{date_stamp}_{output_id}_sample_labelling_metadata.json"
        ),
    )
