"""
Filter job data using deduplicated job ids to create outputs for the analysis pieces
Outputs:
1. A dataset of the deduplicated job metadata
2. A sample of the deduplicated job data with the skills data too
"""

import os
from argparse import ArgumentParser

from ojd_daps_skills import bucket_name
from ojd_daps_skills.getters.data_getters import (
    load_s3_data,
    get_s3_resource,
    save_to_s3,
    load_file,
)

from tqdm import tqdm
import itertools
import pandas as pd
import random


def create_argparser():

    parser = ArgumentParser()

    parser.add_argument(
        "--s3_folder",
        help="S3 folder of data",
        default="escoe_extension/outputs/data/model_application_data",
        type=str,
    )

    parser.add_argument(
        "--local_skills_file_name",
        default="ojd_daps_skills/analysis/OJO/job_ad_to_skills_v2.json",
        type=str,
    )

    parser.add_argument(
        "--dedupe_ids_file_name",
        default="deduplicated_job_ids_6_weeks_v2.csv",
        type=str,
    )

    parser.add_argument("--itl_file_name", default="job_ad_to_itl_v2.csv", type=str)

    parser.add_argument(
        "--occupations_file_name",
        default="raw_job_adverts_additional_fields.csv",
        type=str,
    )
    parser.add_argument(
        "--sample_skills_output", default="dedupe_analysis_skills_sample.json", type=str
    )

    parser.add_argument(
        "--metadata_output", default="dedupe_analysis_metadata.csv", type=str
    )

    return parser


if __name__ == "__main__":

    parser = create_argparser()
    args = parser.parse_args()

    s3 = get_s3_resource()

    # Load data

    # All job skills data
    job_skills = load_file(args.local_skills_file_name, s3=False)

    # The ids of the deduplicated job adverts
    job_ads_deduped = load_s3_data(
        s3, bucket_name, os.path.join(args.s3_folder, args.dedupe_ids_file_name)
    )

    # ITL and occupations datasets
    itl_data = load_s3_data(
        s3, bucket_name, os.path.join(args.s3_folder, args.itl_file_name)
    )
    occ_data = load_s3_data(
        s3, bucket_name, os.path.join(args.s3_folder, args.occupations_file_name)
    )

    # Deduplicate and merge metadata

    merged_metadata = job_ads_deduped.merge(
        itl_data, how="left", left_on="job_id", right_on="id"
    )
    merged_metadata = merged_metadata.merge(
        occ_data, how="left", left_on="job_id", right_on="id"
    )
    merged_metadata.drop(columns=["id_x", "id_y"], inplace=True)
    merged_metadata["job_id"] = merged_metadata["job_id"].astype(str)

    # Get a summary of the number of skills and experiences per job advert
    dedupe_ids = set(job_ads_deduped["job_id"].astype(str).tolist())
    skills_summary = {}
    for job_advert in tqdm(job_skills):
        job_id = job_advert["job_id"]
        if job_id in dedupe_ids:
            num_exp = len(job_advert["skills"].get("EXPERIENCE", []))
            if "SKILL" in job_advert["skills"]:
                skill_ids = [c for a, [b, c] in job_advert["skills"]["SKILL"]]
            else:
                skill_ids = []
            no_match_skill = [s for s in skill_ids if len(s) <= 3]
            matched_skill_ids = [s for s in skill_ids if len(s) > 3]
            skill_lev_ids = [s for s in skill_ids if len(s) > 20]
            skills_summary[job_id] = {
                "num_exp": num_exp,
                "num_skills": len(skill_ids),
                "num_uniq_skills": len(set(skill_ids)),
                "num_uniq_matched_skills": len(set(matched_skill_ids)),
                "num_skill_level": len(skill_lev_ids),
                "num_uniq_skill_level": len(set(skill_lev_ids)),
                "num_no_match": len(no_match_skill),
            }

    skill_meta_names = next(iter(skills_summary.values())).keys()
    for column_name in tqdm(skill_meta_names):
        merged_metadata[column_name] = merged_metadata["job_id"].apply(
            lambda x: skills_summary.get(x, {column_name: None})[column_name]
        )

    print(f"Saving out merged_metadata - {len(merged_metadata)} rows of job adverts")
    save_to_s3(
        s3,
        bucket_name,
        merged_metadata,
        os.path.join(args.s3_folder, args.metadata_output),
    )

    # Sample skills data
    sample_n = 100000
    # Shuffle, then take the first 100000 which are in the dedupe list
    job_skills_shuffled = job_skills.copy()
    random.seed(42)
    random.shuffle(job_skills_shuffled)

    # Cut down this list so as to not have to deal with the computational intensity of the full list
    # Just looking at the first 2*sample_n random shuffled data should be more than enough to get our desired sample size out of
    shorter_job_skills_shuffled = job_skills_shuffled[0 : (2 * sample_n)]
    sampled_job_skills = [
        j for j in shorter_job_skills_shuffled if j["job_id"] in dedupe_ids
    ][0:sample_n]
    sample_job_ids = set([j["job_id"] for j in sampled_job_skills])
    sampled_merged_metadata = merged_metadata[
        merged_metadata["job_id"].isin(sample_job_ids)
    ]

    sampled_merged_metadata["SKILL"] = sampled_merged_metadata["job_id"].map(
        {j["job_id"]: j["skills"].get("SKILL") for j in sampled_job_skills}
    )
    sampled_merged_metadata["EXPERIENCE"] = sampled_merged_metadata["job_id"].map(
        {j["job_id"]: j["skills"].get("EXPERIENCE") for j in sampled_job_skills}
    )
    skills_data_sample = sampled_merged_metadata.to_dict("records")

    print(f"Saving out skills_data_sample - {len(skills_data_sample)} job adverts")

    save_to_s3(
        s3,
        bucket_name,
        skills_data_sample,
        os.path.join(args.s3_folder, args.sample_skills_output),
    )
    save_to_s3(
        s3,
        bucket_name,
        sampled_merged_metadata,
        os.path.join(
            args.s3_folder, args.sample_skills_output.replace(".json", ".csv")
        ),
    )
