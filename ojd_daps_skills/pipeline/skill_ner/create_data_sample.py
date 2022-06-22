"""
This script will output to S3 a random sample of the job adverts.
"""

import random
import os
from datetime import datetime as date

import pandas as pd

from ojd_daps_skills.utils.sql_conn import est_conn
from ojd_daps_skills.getters.data_getters import save_to_s3, get_s3_resource

if __name__ == "__main__":

    BUCKET_NAME = "open-jobs-lake"
    S3_FOLDER = "escoe_extension/inputs/data/skill_ner/data_sample/"
    sample_size = 5000
    conn = est_conn()

    # Get all the job ids in the database
    query_job_ids = "SELECT id " " FROM raw_job_adverts "
    job_ids_df = pd.read_sql(query_job_ids, conn)
    job_ids = job_ids_df["id"].unique().tolist()

    # Randomly sample the job ids
    random.seed(42)
    sampled_job_ids = random.sample(job_ids, min(len(job_ids), sample_size))

    # Get the rest of the job advert data for this sample of adverts
    query_job_descriptions = f"SELECT id, created, job_title_raw, job_location_raw, raw_salary, raw_salary_unit, raw_salary_currency, company_raw, description FROM raw_job_adverts WHERE id IN {tuple(set(sampled_job_ids))}"
    job_ad_sample = pd.read_sql(query_job_descriptions, conn)
    job_ad_sample["created"] = job_ad_sample["created"].astype(str)

    # Save out as dictionary with date stamped preffix
    job_ad_sample_dict = job_ad_sample.set_index("id").T.to_dict("dict")
    date_stamp = str(date.today().date()).replace("-", "")
    s3 = get_s3_resource()
    save_to_s3(
        s3,
        BUCKET_NAME,
        job_ad_sample_dict,
        os.path.join(S3_FOLDER, f"{date_stamp}_sampled_job_ads.json"),
    )
