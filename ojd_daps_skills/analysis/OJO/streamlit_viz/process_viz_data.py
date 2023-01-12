"""
Script to process the skill occurences data into several outputs needed for the Streamlit viz

For each occupation:
- Top 20 most common skills (all skill+groups, just skills group level 0, just skill group level 1, ..2 and 3)
- Top 20 most similar jobs
- Number of job adverts

"""


import os
from collections import Counter
from itertools import chain, combinations
import ast

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from streamlit_agraph import agraph, Node, Edge, Config

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)
from ojd_daps_skills import bucket_name
from ojd_daps_skills.utils.plotting import NESTA_COLOURS


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

    # The ESCO data (ESCO skill code to where in the taxonomy)
    file_name = os.path.join(s3_folder, "skill_ner_mapping", "esco_data_formatted.csv")
    esco_skills = load_s3_data(
        s3,
        bucket_name,
        file_name,
    )
    esco_skills["hierarchy_levels"] = esco_skills["hierarchy_levels"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else None
    )

    return skill_sample, job_id_2_skill_count, skill_id_2_ix, esco_skills


def find_skill_proportions_per_sector(
    skill_sample_df, job_id_2_skill_count, skill_id_2_ix, skill_counts=True
):
    """
    For each sector get the percentage of job adverts each skill is in

    Args:
                    skill_sample_df (DataFrame): A dataframe where each row is a job advert and the skills found are
                                    given in the format outputted by the Skill Extractor package
                    job_id_2_skill_count (dict): A count of each skill found for each job advert (uses an internal skill id)
                            or a list of the skill groups found for each job advert (i.e. no count)
                    skill_id_2_ix (dict): The ESCO skill id to the internal skill id
                    skill_counts (bool): Whether job_id_2_skill_count contains a dict of counts or just a list

    Returns:
                    percentage_sector_skills_df (DataFrame): A dataframe where each row is a sector and each column is a skill
                                    and the values are the percentage of job adverts from this sector which this skill is in

    """

    sector_2_job_ids = skill_sample_df.groupby("sector")["job_id"].unique()

    percentage_sector_skills = {}
    for sector, job_ids in tqdm(sector_2_job_ids.items()):
        total_sector_skills = Counter()
        for job_id in job_ids:
            if skill_counts:
                skill_names = job_id_2_skill_count[job_id].keys()
            else:
                skill_names = job_id_2_skill_count[job_id]
            total_sector_skills += Counter(skill_names)
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

    return sector_2_job_ids, percentage_sector_skills_df


def get_skill_levels(s):
    """
    Args:
            s (str): An ESCO skill group code from any location (e.g S3.1.2, K1099, S4)
    Returns:
            The break up of where this code is in the taxonomy, S, S3, S3.1, S3.1.2
    """
    if "K" in s:
        if len(s) == 5:
            lev_3 = s
            lev_2 = s[0:4]
            lev_1 = s[0:3]
        else:
            lev_3 = None
            if len(s) == 4:
                lev_2 = s
                lev_1 = s[0:3]
            else:
                lev_2 = None
                lev_1 = s

    elif "L" in s:
        lev_3 = None
        lev_2 = None
        lev_1 = s[0:2]
    else:
        split_by_dot = s.split(".")
        if len(split_by_dot) == 3:
            lev_3 = s
            lev_2 = ".".join(split_by_dot[0:2])
            lev_1 = ".".join(split_by_dot[0:1])
        elif len(split_by_dot) == 2:
            lev_3 = None
            lev_2 = s
            lev_1 = ".".join(split_by_dot[0:1])
        elif len(split_by_dot) == 1:
            lev_3 = None
            lev_2 = None
            lev_1 = s
        else:
            lev_1 = s
    return s[0], lev_1, lev_2, lev_3


def get_skill_per_taxonomy_level(esco_skills, job_id_2_skill_count, skill_id_2_ix):

    ix_2_skill_id = {str(v): k for k, v in skill_id_2_ix.items()}

    skill_esco_skills = esco_skills[esco_skills["id"].apply(lambda x: len(str(x)) > 10)]
    skill_id_2_levels = dict(
        zip(skill_esco_skills["id"], skill_esco_skills["hierarchy_levels"])
    )

    job_id_2_skill_hier_mentions_per_lev = {}
    for job_id, skill_counts in tqdm(job_id_2_skill_count.items()):

        job_ad_skills = [ix_2_skill_id[ix] for ix in list(skill_counts.keys())]

        job_lev_0 = set()
        job_lev_1 = set()
        job_lev_2 = set()
        job_lev_3 = set()
        for skill_id in job_ad_skills:
            if len(skill_id) > 10:
                hier_levels = skill_id_2_levels.get(skill_id)
                if hier_levels:
                    for hier_level in hier_levels:
                        job_lev_0.add(hier_level[0])
                        lev_1 = hier_level[1]
                        lev_2 = hier_level[2]
                        lev_3 = hier_level[3]
                        if lev_1:
                            job_lev_1.add(lev_1)
                        if lev_2:
                            job_lev_2.add(lev_2)
                        if lev_3:
                            job_lev_3.add(lev_3)
            else:
                lev_0, lev_1, lev_2, lev_3 = get_skill_levels(skill_id)
                if lev_1:
                    job_lev_1.add(lev_1)
                if lev_2:
                    job_lev_2.add(lev_2)
                if lev_3:
                    job_lev_3.add(lev_3)

        job_id_2_skill_hier_mentions_per_lev[job_id] = {
            "0": job_lev_0,
            "1": job_lev_1,
            "2": job_lev_2,
            "3": job_lev_3,
        }

    return job_id_2_skill_hier_mentions_per_lev


def get_top_skills_per_sector(percentage_sector_skills_df, esco_code2name, top_n=20):
    top_skills_per_sector = {}
    for sector_name, sector_skill_percentages in percentage_sector_skills_df.iterrows():
        top_skills_per_sector[sector_name] = {
            esco_code2name.get(skill_id, skill_id): top_skills
            for skill_id, top_skills in sector_skill_percentages.sort_values(
                ascending=False
            )[0:top_n]
            .to_dict()
            .items()
        }
    return top_skills_per_sector


if __name__ == "__main__":

    s3 = get_s3_resource()

    s3_folder = "escoe_extension/outputs/data"

    skill_sample, job_id_2_skill_count, skill_id_2_ix, esco_skills = load_datasets(
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

    esco_code2name["K"] = "Knowledge"
    esco_code2name["S"] = "Skills"
    esco_code2name["T"] = "Transversal skills and competencies"
    esco_code2name["A"] = "Attitudes"
    esco_code2name["L"] = "Language skills and Knowledge"

    top_n = 100

    sector_2_job_ids, percentage_sector_skills_df = find_skill_proportions_per_sector(
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

    top_skills_per_sector = get_top_skills_per_sector(
        percentage_sector_skills_df, esco_code2name, top_n=20
    )

    number_job_adverts_per_sector = (
        skill_sample_df.groupby("sector")["job_id"].count().to_dict()
    )

    job_id_2_skill_hier_mentions_per_lev = get_skill_per_taxonomy_level(
        esco_skills, job_id_2_skill_count, skill_id_2_ix
    )

    percentage_sector_skill_by_group_list = []
    for group_num in ["0", "1", "2", "3"]:
        _, percentage_sector_skill_by_group = find_skill_proportions_per_sector(
            skill_sample_df,
            {
                job_id: skills[group_num]
                for job_id, skills in job_id_2_skill_hier_mentions_per_lev.items()
            },
            skill_id_2_ix,
            skill_counts=False,
        )
        top_skill_by_group_per_sector = get_top_skills_per_sector(
            percentage_sector_skill_by_group, esco_code2name, top_n=20
        )
        percentage_sector_skill_by_group_list.append(top_skill_by_group_per_sector)

    # Combine all sector data together
    all_sector_data = {}
    for sector_name, num_ads in number_job_adverts_per_sector.items():
        all_sector_data[sector_name] = {
            "similar_sectors": similar_sectors_per_sector[sector_name],
            "num_ads": num_ads,
            "top_skills": {
                "all": top_skills_per_sector[sector_name],
                "0": percentage_sector_skill_by_group_list[0][sector_name],
                "1": percentage_sector_skill_by_group_list[1][sector_name],
                "2": percentage_sector_skill_by_group_list[2][sector_name],
                "3": percentage_sector_skill_by_group_list[3][sector_name],
            },
        }

    save_to_s3(
        s3,
        bucket_name,
        all_sector_data,
        os.path.join(s3_folder, "streamlit_viz", "per_sector_sample.json"),
    )

    # Network data

    # Find the average skill percentages per sector
    average_sector_skills = {}
    for sector, job_ids in tqdm(sector_2_job_ids.items()):
        total_sector_skills = Counter()
        for job_id in job_ids:
            total_sector_skills += Counter(job_id_2_skill_count[job_id])
        average_sector_skills[sector] = {
            k: v / len(job_ids) for k, v in total_sector_skills.items()
        }

    average_sector_skills_df = pd.DataFrame(average_sector_skills)
    average_sector_skills_df = average_sector_skills_df.T
    average_sector_skills_df.fillna(value=0, inplace=True)

    field_name_2_index = {
        field: n for n, field in enumerate(average_sector_skills_df.index)
    }

    dists_between_sectors = euclidean_distances(
        average_sector_skills_df, average_sector_skills_df
    )

    def get_euc_dist(source, target, dists, field_name_2_index):
        return dists[field_name_2_index[str(source)], field_name_2_index[str(target)]]

    pairs = list(
        combinations(sorted(list(set(average_sector_skills_df.index.tolist()))), 2)
    )
    pairs = [x for x in pairs if len(x) > 0]
    edge_list = pd.DataFrame(pairs, columns=["source", "target"])
    edge_list["weight"] = edge_list.apply(
        lambda x: get_euc_dist(
            x.source, x.target, dists_between_sectors, field_name_2_index
        ),
        axis=1,
    )
    edge_list["weight"] = edge_list["weight"].apply(
        lambda x: 1 / (x + 0.000001)
    )  # Because a lower euclide is a higher weighting

    # Set weight to be between 0 and 1 so it's more clear

    min_weight = edge_list["weight"].min()
    weight_diff = edge_list["weight"].max() - min_weight
    edge_list["weight"] = edge_list["weight"].apply(
        lambda x: (x - min_weight) / weight_diff
    )

    save_to_s3(
        s3,
        bucket_name,
        edge_list,
        os.path.join(
            s3_folder, "streamlit_viz", "skill_similarity_between_sectors_sample.csv"
        ),
    )

    # Get sector to knowledge domain mapper
    sector_2_kd = dict(
        zip(skill_sample_df["sector"], skill_sample_df["knowledge_domain"])
    )

    save_to_s3(
        s3,
        bucket_name,
        sector_2_kd,
        os.path.join(s3_folder, "streamlit_viz", "sector_2_kd_sample.json"),
    )

    # Lightweight edge_list - only with sectors which have a decent number of job adverts
    top_sectors = set([k for k, v in all_sector_data.items() if v["num_ads"] > 100])

    edge_list_lightweight = edge_list[
        (edge_list["source"].isin(top_sectors) & edge_list["target"].isin(top_sectors))
    ]

    save_to_s3(
        s3,
        bucket_name,
        edge_list_lightweight,
        os.path.join(
            s3_folder,
            "streamlit_viz",
            "lightweight_skill_similarity_between_sectors_sample.csv",
        ),
    )
