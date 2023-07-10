"""
Process a dataset of job adverts for labelling in Prodigy

This includes formatting the random sample of 100,000 (mixed green and brown) job adverts
created for the green jobs project https://github.com/nestauk/dap_prinz_green_jobs
"""

import pandas as pd
import boto3

import json
import random

from ojd_daps_skills.getters.data_getters import load_s3_data, get_s3_resource
from ojd_daps_skills.pipeline.skill_ner.ner_spacy_utils import detect_camelcase


def clean_text(text):
    text = text.encode("ascii", "ignore").decode()
    text = detect_camelcase(text)
    return text


if __name__ == "__main__":

    s3 = get_s3_resource()

    jobs_sample = load_s3_data(
        s3,
        "prinz-green-jobs",
        "outputs/data/ojo_application/deduplicated_sample/mixed_ojo_sample.csv",
    )

    output_file_dir = "escoe_extension/outputs/labelled_job_adverts/prodigy/processed_sample_20230710.jsonl"

    jobs_sample = jobs_sample[pd.notnull(jobs_sample["description"])]
    jobs_sample.loc[:, "description"] = jobs_sample["description"].apply(clean_text)

    jobs_sample["meta"] = jobs_sample[["id"]].to_dict(orient="records")
    jobs_sample_formated = (
        jobs_sample[["description", "meta"]]
        .rename(columns={"description": "text"})
        .to_dict(orient="records")
    )

    # We aren't going to be able to label all 100,000 of the sample, so cut it down
    random.seed(42)
    jobs_sample_formated = random.sample(jobs_sample_formated, 5000)

    output_string = ""

    for line in jobs_sample_formated:
        output_string += json.dumps(line, ensure_ascii=False)
        output_string += "\n"

    s3 = boto3.client("s3")
    s3.put_object(Body=output_string, Bucket="open-jobs-lake", Key=output_file_dir)
