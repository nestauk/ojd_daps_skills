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
from ojd_daps_skills import bucket_name, config, logger

from datetime import datetime as date
import json
import os

if __name__ == "__main__":

    s3 = get_s3_resource()

    file_name = os.path.join(config["s3_ner_folder"], config["sample_file_name"])
    logger.info(f"Processing job advert sample from {file_name}")

    sample_data = load_s3_data(s3, bucket_name, file_name)
    date_stamp = str(date.today().date()).replace("-", "")

    s3_label_output_folder = config["s3_label_output_folder"]

    output_id = 0
    line_index = 0
    texts = []
    index_metadata = {}
    for job_id, job_info in sample_data.items():
        # If there are any description texts with '\n' in this will
        # mess the sentence separation up in the output step,
        # so just make sure they are all removed.
        texts.append(job_info["description"].replace("\n", "."))
        index_metadata[line_index] = job_id
        line_index += 1
        if line_index == 400:
            # Output to S3
            save_to_s3(
                s3,
                bucket_name,
                "\n".join(texts),
                os.path.join(
                    s3_label_output_folder,
                    f"{date_stamp}_{output_id}_sample_labelling_text_data.txt",
                ),
            )
            save_to_s3(
                s3,
                bucket_name,
                index_metadata,
                os.path.join(
                    s3_label_output_folder,
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
        bucket_name,
        "\n".join(texts),
        os.path.join(
            s3_label_output_folder,
            f"{date_stamp}_{output_id}_sample_labelling_text_data.txt",
        ),
    )
    save_to_s3(
        s3,
        bucket_name,
        index_metadata,
        os.path.join(
            s3_label_output_folder,
            f"{date_stamp}_{output_id}_sample_labelling_metadata.json",
        ),
    )
