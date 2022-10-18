"""
One off script to filter the output results for just the sample of job adverts.
The saved out files will be easier to load for any analysis.
"""

import pandas as pd

from ojd_daps_skills.getters.data_getters import (
    save_to_s3,
    get_s3_resource,
    get_s3_data_paths,
    load_s3_json,
    load_s3_data,
    load_file,
)
from ojd_daps_skills import bucket_name

s3 = get_s3_resource()

# Load job adverts sample. First row was in the header.
file_name = (
    "escoe_extension/outputs/data/model_application_data/raw_job_adverts_sample.csv"
)
obj = s3.Object(bucket_name, file_name)
raw_job_adverts = pd.read_csv("s3://" + bucket_name + "/" + file_name, header=None)
raw_job_adverts.rename(
    columns={0: "job_id", 1: "date", 2: "job_title", 3: "job_ad"}, inplace=True
)
len(raw_job_adverts)

# The job advert ids of those in the sample
raw_job_adverts_ids = set(raw_job_adverts["job_id"].unique().tolist())

itl_file_name = "escoe_extension/outputs/data/model_application_data/job_ad_to_itl.csv"
full_itl_data = load_s3_data(s3, bucket_name, itl_file_name)
full_itl_data_sample = full_itl_data[
    full_itl_data["id"].isin(raw_job_adverts_ids)
].reset_index(drop=True)
save_to_s3(
    s3,
    bucket_name,
    full_itl_data_sample,
    "escoe_extension/outputs/data/model_application_data/job_ad_to_itl_sample.csv",
)

add_fields_file_name = "escoe_extension/outputs/data/model_application_data/raw_job_adverts_additional_fields.csv"
add_fields = load_s3_data(s3, bucket_name, add_fields_file_name)
add_fields_sample = add_fields[add_fields["id"].isin(raw_job_adverts_ids)].reset_index(
    drop=True
)
save_to_s3(
    s3,
    bucket_name,
    add_fields_sample,
    "escoe_extension/outputs/data/model_application_data/raw_job_adverts_additional_fields_sample.csv",
)

duplicates_file_name = (
    "escoe_extension/outputs/data/model_application_data/job_ad_duplicates.csv"
)
duplicates = load_s3_data(s3, bucket_name, duplicates_file_name)
duplicates_sample = duplicates[
    duplicates["first_id"].isin(raw_job_adverts_ids)
].reset_index(drop=True)
save_to_s3(
    s3,
    bucket_name,
    duplicates_sample,
    "escoe_extension/outputs/data/model_application_data/job_ad_duplicates_sample.csv",
)

skills_file_name = "ojd_daps_skills/utils/job_ad_to_skills.json"  # Needed to be downloaded from S3 and saved locally since its so big
skills_data = load_file(skills_file_name, s3=False)
skills_data_sample = [s for s in skills_data if int(s["job_id"]) in raw_job_adverts_ids]
save_to_s3(
    s3,
    bucket_name,
    skills_data_sample,
    "escoe_extension/outputs/data/model_application_data/job_ad_to_skills_sample.json",
)
