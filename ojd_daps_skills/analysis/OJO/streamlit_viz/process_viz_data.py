"""
Script to process the skill occurences data into several outputs needed for the Streamlit viz

For each occupation:
- Top 100 most common skills
- Top 100 most similar jobs
- Number of job adverts

"""


import os
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)
from ojd_daps_skills import bucket_name


def clean_sector_name(sector_name):

    return sector_name.replace("&amp;", "&")


def load_datasets(s3, s3_folder, bucket_name):
    """
    Load all the neccessary datasets from S3
    """

    # The skill sample (with metadata)
    file_name = os.path.join(
        s3_folder,
        "model_application_data",
        "dedupe_analysis_skills_sample_temp_fix.json",
    )
    skill_sample = load_s3_data(s3, bucket_name, file_name)

    # The skill occurences
    file_name = os.path.join(
        s3_folder,
        "analysis",
        "job_ad_to_mapped_skills_occurrences_sample_30112022.json",
    )
    job_id_2_skill_count = load_s3_data(s3, bucket_name, file_name)

    # The esco skill to ix mapper
    file_name = os.path.join(
        s3_folder, "analysis", "mapped_skills_index_dict_30112022.json"
    )
    skill_id_2_ix = load_s3_data(s3, bucket_name, file_name)
    skill_id_2_ix = {k: str(v) for k, v in skill_id_2_ix.items()}

    return skill_sample, job_id_2_skill_count, skill_id_2_ix


def find_skill_proportions_per_sector(
    skill_sample_df, job_id_2_skill_count, skill_id_2_ix
):
    """
    For each sector get the percentage of job adverts each skill is in

    Args:
            skill_sample_df (DataFrame): A dataframe where each row is a job advert and the skills found are
                    given in the format outputted by the Skill Extractor package
            job_id_2_skill_count (dict): A count of each skill found for each job advert (uses an internal skill id)
            skill_id_2_ix (dict): The ESCO skill id to the internal skill id

    Returns:
            percentage_sector_skills_df (DataFrame): A dataframe where each row is a sector and each column is a skill
                    and the values are the percentage of job adverts from this sector which this skill is in

    """

    sector_2_job_ids = skill_sample_df.groupby("sector")["job_id"].unique()

    percentage_sector_skills = {}
    for sector, job_ids in tqdm(sector_2_job_ids.items()):
        total_sector_skills = Counter()
        for job_id in job_ids:
            total_sector_skills += Counter(job_id_2_skill_count[job_id].keys())
        percentage_sector_skills[sector] = {
            k: v / len(job_ids) for k, v in total_sector_skills.items()
        }

    percentage_sector_skills_df = pd.DataFrame(percentage_sector_skills)
    percentage_sector_skills_df = percentage_sector_skills_df.T
    percentage_sector_skills_df.fillna(value=0, inplace=True)
    # Map column names back to their ESCO codes
    percentage_sector_skills_df.rename(
        columns={v: k for k, v in skill_id_2_ix.items()}, inplace=True
    )

    return percentage_sector_skills_df


if __name__ == "__main__":

    s3 = get_s3_resource()

    s3_folder = "escoe_extension/outputs/data"

    skill_sample, job_id_2_skill_count, skill_id_2_ix = load_datasets(
        s3, s3_folder, bucket_name
    )

    skill_sample_df = pd.DataFrame(skill_sample)
    skill_sample_df["sector"] = skill_sample_df["sector"].apply(clean_sector_name)

    # Use the skill sample to get all the ESCO code to ESCO names
    esco_code2name = {
        c: b
        for job_skills in skill_sample
        if job_skills.get("SKILL")
        for a, [b, c] in job_skills.get("SKILL", [None, [None, None]])
    }

    top_n = 100

    percentage_sector_skills_df = find_skill_proportions_per_sector(
        skill_sample_df, job_id_2_skill_count, skill_id_2_ix
    )

    dists = euclidean_distances(
        percentage_sector_skills_df.values, percentage_sector_skills_df.values
    )

    similar_sectors_per_sector = {}
    for i, sector_name in enumerate(percentage_sector_skills_df.index):
        most_common_ix = np.argpartition(dists[i], top_n)[0 : (top_n + 1)]
        similar_sectors_per_sector[sector_name] = {
            percentage_sector_skills_df.index[ix]: dists[i][ix]
            for ix in most_common_ix
            if ix != i
        }

    top_skills_per_sector = {}
    for sector_name, sector_skill_percentages in percentage_sector_skills_df.iterrows():
        top_skills_per_sector[sector_name] = {
            esco_code2name[skill_id]: top_skills
            for skill_id, top_skills in sector_skill_percentages.sort_values(
                ascending=False
            )[0:top_n]
            .to_dict()
            .items()
        }

    number_job_adverts_per_sector = (
        skill_sample_df.groupby("sector")["job_id"].count().to_dict()
    )

    save_to_s3(
        s3,
        bucket_name,
        similar_sectors_per_sector,
        os.path.join(
            s3_folder, "streamlit_viz", "similar_sectors_per_sector_sample.json"
        ),
    )
    save_to_s3(
        s3,
        bucket_name,
        top_skills_per_sector,
        os.path.join(s3_folder, "streamlit_viz", "top_skills_per_sector_sample.json"),
    )
    save_to_s3(
        s3,
        bucket_name,
        number_job_adverts_per_sector,
        os.path.join(
            s3_folder, "streamlit_viz", "number_job_adverts_per_sector_sample.json"
        ),
    )
