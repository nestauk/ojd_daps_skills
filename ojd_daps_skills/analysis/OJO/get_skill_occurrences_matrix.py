import os
from datetime import date
from collections import Counter

import pandas as pd
from tqdm import tqdm

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)
from ojd_daps_skills import bucket_name

s3 = get_s3_resource()


def get_cooccurence_matrix(job_id_2_skill_count, skill_id_2_ix, convert_int=True):
    # Convert dicts to cooccurrence matrix
    job_id_2_skill_count_df = pd.DataFrame(job_id_2_skill_count)
    job_id_2_skill_count_df = job_id_2_skill_count_df.T
    job_id_2_skill_count_df.fillna(value=0, inplace=True)
    if convert_int:
        job_id_2_skill_count_df = job_id_2_skill_count_df.astype(int)
    # Map column names back to their ESCO codes
    job_id_2_skill_count_df.rename(
        columns={v: k for k, v in skill_id_2_ix.items()}, inplace=True
    )

    return job_id_2_skill_count_df


if __name__ == "__main__":

    s3_folder = "escoe_extension/outputs/data/model_application_data"

    # Get todays date for the output name prefix
    today = date.today().strftime("%d%m%Y")

    # Load the skill sample
    file_name = os.path.join(s3_folder, "dedupe_analysis_skills_sample_temp_fix.json")
    skill_sample = load_s3_data(s3, bucket_name, file_name)

    # Find all the ESCO skill codes
    all_skill_codes = set()
    for job_advert in tqdm(skill_sample):
        if job_advert["SKILL"]:
            job_skills = [s[1][1] for s in job_advert["SKILL"]]
            all_skill_codes.update(set(job_skills))

    # Creater a mapper from ESCO skill code to an index (will help with processing time)
    # {'7d10fcb2-b368-48ab-996b-7c9fafcf68ed': 0, 'dce16d2c-278a-4161-9847-8435e52c96d3': 1,...}
    skill_id_2_ix = dict(zip(all_skill_codes, range(len(all_skill_codes))))

    # Get count of each skill in each job advert
    job_id_2_skill_count = {}
    for job_advert in tqdm(skill_sample):
        if job_advert["SKILL"]:
            job_skills = [skill_id_2_ix[s[1][1]] for s in job_advert["SKILL"]]
            job_id_2_skill_count[job_advert["job_id"]] = dict(Counter(job_skills))
        else:
            job_id_2_skill_count[job_advert["job_id"]] = {}

    # Save out
    save_to_s3(
        s3,
        bucket_name,
        job_id_2_skill_count,
        f"escoe_extension/outputs/data/analysis/job_ad_to_mapped_skills_occurrences_sample_{today}.json",
    )

    save_to_s3(
        s3,
        bucket_name,
        skill_id_2_ix,
        f"escoe_extension/outputs/data/analysis/mapped_skills_index_dict_{today}.json",
    )

    print("Calculating and saving matrix")
    job_id_2_skill_count_df = get_cooccurence_matrix(
        job_id_2_skill_count, skill_id_2_ix
    )

    # This is big!
    save_to_s3(
        s3,
        bucket_name,
        job_id_2_skill_count_df,
        f"escoe_extension/outputs/data/analysis/job_ad_to_mapped_skills_occurrences_sample_matrix_{today}.csv",
    )
