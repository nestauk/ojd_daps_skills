"""
Process a dataset of job adverts for labelling in Prodigy

This includes formatting the random sample of 100,000 (mixed green and brown) job adverts
created for the green jobs project https://github.com/nestauk/dap_prinz_green_jobs
"""

import pandas as pd
import boto3

import json
import random
import re

from ojd_daps_skills.getters.data_getters import load_s3_data, get_s3_resource
from ojd_daps_skills.pipeline.skill_ner.ner_spacy_utils import detect_camelcase

punctuation_replacement_rules = {
    # old patterns: replacement pattern
    # Convert bullet points to fullstops
    "[\u2022\u2023\u25E6\u2043\u2219*]": ".",
    r"[/:\\]": " ",  # Convert colon and forward and backward slashes to spaces
}

compiled_punct_patterns = {
    re.compile(p): v for p, v in punctuation_replacement_rules.items()
}


def pad_out_punct(text):
    # When punctuation is directly followed by a letter, then pad it out with a space
    pattern = r"([,?.)!;:])([a-zA-Z])"
    replacement = r"\1 \2"
    result = re.sub(pattern, replacement, text)
    return result


def replacements(text):
    """
    Ampersands and bullet points need some tweaking to be most useful in the pipeline.
    Some job adverts have different markers for a bullet pointed list. When this happens
    we want them to be in a fullstop separated format.
    e.g. ";• managing the grants database;• preparing financial and interna"
    ":•\xa0NMC registration paid every year•\xa0Free train"
    """
    text = (
        text.replace("&", "and")
        .replace("\xa0", " ")
        .replace("\n", ".")
        .replace("[", "")
        .replace("]", "")
    )

    for pattern, rep in compiled_punct_patterns.items():
        text = pattern.sub(rep, text)

    text = pad_out_punct(text)

    return text.strip()


def clean_text(text):
    text = text.encode("ascii", "ignore").decode()
    text = detect_camelcase(text)
    text = replacements(text)

    # clean up all multiple spaces
    text = " ".join(text.split())

    return text


if __name__ == "__main__":

    s3 = get_s3_resource()

    jobs_sample = load_s3_data(
        s3,
        "prinz-green-jobs",
        "outputs/data/ojo_application/deduplicated_sample/mixed_ojo_sample.csv",
    )

    output_file_dir = "escoe_extension/outputs/labelled_job_adverts/prodigy/processed_sample_20230801.jsonl"

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
