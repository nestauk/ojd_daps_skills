"""
Script to process the skill occurences data into several outputs needed for the Streamlit viz
For each occupation and regions:
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
from collections import defaultdict

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)
from ojd_daps_skills import bucket_name, logger
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


def find_skill_proportions_per_group(
    skill_sample_df, job_id_2_skill_count, skill_id_2_ix, group, skill_counts=True
):
    """
    For each group get the percentage of job adverts each skill is in
    Args:
                    skill_sample_df (DataFrame): A dataframe where each row is a job advert and the skills found are
                                    given in the format outputted by the Skill Extractor package
                    job_id_2_skill_count (dict): A count of each skill found for each job advert (uses an internal skill id)
                            or a list of the skill groups found for each job advert (i.e. no count)
                    skill_id_2_ix (dict): The ESCO skill id to the internal skill id
                    skill_counts (bool): Whether job_id_2_skill_count contains a dict of counts or just a list
    Returns:
                    percentage_group_skills_df (DataFrame): A dataframe where each row is a group and each column is a skill
                                    and the values are the percentage of job adverts from this group which this skill is in
    """

    group_2_job_ids = skill_sample_df.groupby(group)["job_id"].unique()

    percentage_group_skills = {}
    for selected_group, job_ids in tqdm(group_2_job_ids.items()):
        total_group_skills = Counter()
        for job_id in job_ids:
            if skill_counts:
                skill_names = job_id_2_skill_count[job_id].keys()
            else:
                skill_names = job_id_2_skill_count[job_id]
            total_group_skills += Counter(skill_names)
        percentage_group_skills[selected_group] = {
            k: v / len(job_ids) for k, v in total_group_skills.items()
        }

    percentage_group_skills_df = pd.DataFrame(percentage_group_skills).T
    percentage_group_skills_df.fillna(value=0, inplace=True)
    percentage_group_skills_df.rename(
        columns={v: k for k, v in skill_id_2_ix.items()}, inplace=True
    )

    return group_2_job_ids, percentage_group_skills_df


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
        job_skill_level = set()
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
                job_skill_level.add(skill_id)
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
            "4": job_skill_level,
        }

    return job_id_2_skill_hier_mentions_per_lev


def get_top_skills_per_group(
    percentage_group_skills_df, esco_code2name, top_n=20, esco_id_2_trans_flag=None
):
    top_skills_per_group = {}
    for group_name, group_skill_percentages in percentage_group_skills_df.iterrows():
        if not esco_id_2_trans_flag:
            top_skills_per_group[group_name] = {
                esco_code2name.get(skill_id, skill_id): top_skills
                for skill_id, top_skills in group_skill_percentages.sort_values(
                    ascending=False
                )[0:top_n]
                .to_dict()
                .items()
            }
        else:
            # Don't include transversal skills
            top_skills_per_group[group_name] = {
                esco_code2name.get(skill_id, skill_id): top_skills
                for skill_id, top_skills in group_skill_percentages.sort_values(
                    ascending=False
                )[0:top_n]
                .to_dict()
                .items()
                if not esco_id_2_trans_flag.get(skill_id, False)
            }
    return top_skills_per_group


def get_only_top_transversal_skills_per_group(
    percentage_group_skills_df,
    esco_code2name,
    esco_id_2_trans_flag,
    top_n=20,
):
    top_skills_per_group = {}
    for group_name, group_skill_percentages in percentage_group_skills_df.iterrows():
        all_trans_skills = {
            esco_code2name.get(skill_id, skill_id): top_skills
            for skill_id, top_skills in group_skill_percentages
            .to_dict()
            .items()
            if esco_id_2_trans_flag.get(skill_id)
        }
        sorted_trans_skills = dict(
            sorted(all_trans_skills.items(), key=lambda item: item[1], reverse=True)
        )
        top_skills_per_group[group_name] = {
            k: sorted_trans_skills[k] for k in list(sorted_trans_skills)[:top_n]
        }

    return top_skills_per_group


def esco_id_label(esco_id, hier_levels, search_for="T", just_skill_level=False):
    if len(esco_id) > 10:
        if search_for in str(hier_levels):
            return True
        else:
            return False
    else:
        if just_skill_level:
            return False
        else:
            if search_for in str(esco_id):
                return True
            else:
                return False


def get_most_common_skills(
    job_id_2_skill_count, skill_id_2_ix, esco_code2name, esco_skills, top_n=20
):

    # For every skill - the proportion of job adverts its in
    num_job_ads = len(job_id_2_skill_count)
    skill_num_ads = Counter()
    for skill_counts in tqdm(job_id_2_skill_count.values()):
        skill_num_ads += Counter(skill_counts.keys())
    skill_prop_ads = {k: v / num_job_ads for k, v in skill_num_ads.items()}

    skill_prop_ads_df = pd.DataFrame.from_dict(
        skill_prop_ads, orient="index", columns=["prop_job_ads"]
    )
    skill_prop_ads_df["esco_code"] = skill_prop_ads_df.index.map(
        {v: k for k, v in skill_id_2_ix.items()}
    )

    # Separate by skills in the S1, S2, S3, ..., T parts of the taxonomy and output out the most common skills
    # (at the skill level). Also output the most common skills from any level.
    top_skills_by_skill_groups = {}
    s_level_codes = [
        esco_code
        for esco_code in esco_code2name.keys()
        if esco_code[0] == "S" and len(esco_code) in range(2, 4)
    ]
    for search_for_code in tqdm(s_level_codes + ["T", "all"]):
        if search_for_code != "all":
            skill_group_ids = set(
                esco_skills[
                    esco_skills.apply(
                        lambda x: esco_id_label(
                            x["id"],
                            x["hierarchy_levels"],
                            search_for=search_for_code,
                            just_skill_level=True,
                        ),
                        axis=1,
                    )
                ]["id"].tolist()
            )
            skill_group_props = skill_prop_ads_df[
                skill_prop_ads_df["esco_code"].isin(skill_group_ids)
            ]
            dict_key_name = f"{esco_code2name[search_for_code]} ({search_for_code})"
        else:
            skill_group_props = skill_prop_ads_df[
                skill_prop_ads_df["esco_code"].str.len() > 10
            ]
            dict_key_name = search_for_code
        top_skills_group = skill_group_props.sort_values(
            by="prop_job_ads", ascending=False
        )[0:top_n]
        top_skills_group["esco_name"] = top_skills_group["esco_code"].map(
            esco_code2name
        )
        top_skills_by_skill_groups[dict_key_name] = dict(
            zip(top_skills_group["esco_name"], top_skills_group["prop_job_ads"])
        )

    return top_skills_by_skill_groups


if __name__ == "__main__":

    logger.info("processing data!")

    s3 = get_s3_resource()

    s3_folder = "escoe_extension/outputs/data"

    skill_sample, job_id_2_skill_count, skill_id_2_ix, esco_skills = load_datasets(
        s3, s3_folder, bucket_name
    )

    esco_skills["is_transversal"] = esco_skills.apply(
        lambda x: esco_id_label(x["id"], x["hierarchy_levels"], search_for="T"), axis=1
    )
    esco_id_2_trans_flag = dict(zip(esco_skills["id"], esco_skills["is_transversal"]))

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

    # here add the rest of K skills
    k_skills = (
        esco_skills.query('id.str.startswith("K")')
        .set_index("id")
        .description.to_dict()
    )
    esco_code2name.update(k_skills)

    # here add the rest of S skills

    s_skills = (
        esco_skills.query('id.str.startswith("S")')
        .set_index("id")
        .description.to_dict()
    )
    esco_code2name.update(s_skills)

    top_skills_by_skill_groups = get_most_common_skills(
        job_id_2_skill_count, skill_id_2_ix, esco_code2name, esco_skills, top_n=20
    )

    save_to_s3(
        s3,
        bucket_name,
        top_skills_by_skill_groups,
        os.path.join(
            s3_folder, "streamlit_viz", "per_skill_group_proportions_sample.json"
        ),
    )

    top_n = 100

    sector_2_job_ids, percentage_sector_skills_df = find_skill_proportions_per_group(
        skill_sample_df, job_id_2_skill_count, skill_id_2_ix, group="sector"
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

    top_skills_per_sector = get_top_skills_per_group(
        percentage_sector_skills_df, esco_code2name, top_n=20
    )
    top_skills_per_sector_no_trans = get_top_skills_per_group(
        percentage_sector_skills_df,
        esco_code2name,
        top_n=20,
        esco_id_2_trans_flag=esco_id_2_trans_flag,
    )

    top_trans_skills_per_sector = get_only_top_transversal_skills_per_group(
        percentage_sector_skills_df,
        esco_code2name,
        top_n=20,
        esco_id_2_trans_flag=esco_id_2_trans_flag,
    )

    number_job_adverts_per_sector = (
        skill_sample_df.groupby("sector")["job_id"].count().to_dict()
    )

    job_id_2_skill_hier_mentions_per_lev = get_skill_per_taxonomy_level(
        esco_skills, job_id_2_skill_count, skill_id_2_ix
    )
    percentage_sector_skill_by_group_list = []
    percentage_sector_skill_by_group_list_no_trans = []
    percentage_sector_trans_skill_by_group = []
    for group_num in ["0", "1", "2", "3", "4"]:
        _, percentage_sector_skill_by_group = find_skill_proportions_per_group(
            skill_sample_df,
            {
                job_id: skills[group_num]
                for job_id, skills in job_id_2_skill_hier_mentions_per_lev.items()
            },
            skill_id_2_ix,
            skill_counts=False,
            group="sector",
        )
        top_skill_by_group_per_sector = get_top_skills_per_group(
            percentage_sector_skill_by_group, esco_code2name, top_n=20
        )
        top_skill_by_group_per_sector_no_trans = get_top_skills_per_group(
            percentage_sector_skill_by_group,
            esco_code2name,
            top_n=20,
            esco_id_2_trans_flag=esco_id_2_trans_flag,
        )
        top_skill_per_group_only_trans = get_only_top_transversal_skills_per_group(
            percentage_sector_skill_by_group,
            esco_code2name,
            top_n=20,
            esco_id_2_trans_flag=esco_id_2_trans_flag,
        )
        percentage_sector_skill_by_group_list.append(top_skill_by_group_per_sector)
        percentage_sector_skill_by_group_list_no_trans.append(
            top_skill_by_group_per_sector_no_trans
        )
        percentage_sector_trans_skill_by_group.append(top_skill_per_group_only_trans)

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
                "4": percentage_sector_skill_by_group_list[4][sector_name],
            },
            "top_skills_no_transversal": {
                "all": top_skills_per_sector_no_trans[sector_name],
                "0": percentage_sector_skill_by_group_list_no_trans[0][sector_name],
                "1": percentage_sector_skill_by_group_list_no_trans[1][sector_name],
                "2": percentage_sector_skill_by_group_list_no_trans[2][sector_name],
                "3": percentage_sector_skill_by_group_list_no_trans[3][sector_name],
                "4": percentage_sector_skill_by_group_list_no_trans[4][sector_name],
            },
            # as transversal skills appear to be either level 2 or at the skill level 
            "top_transversal_skills": {
                "all": top_trans_skills_per_sector[sector_name],
                "2": percentage_sector_trans_skill_by_group[2][sector_name],
                "4": percentage_sector_trans_skill_by_group[4][sector_name],
            },
        }
    # Get sector to knowledge domain mapper
    sector_2_kd = dict(
        zip(skill_sample_df["sector"], skill_sample_df["knowledge_domain"])
    )
    save_to_s3(
        s3,
        bucket_name,
        all_sector_data,
        os.path.join(s3_folder, "streamlit_viz", "per_sector_sample_updated.json"),
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

    # Lightweight edge_list - only with sectors which have a decent number of
    # job adverts
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

    # LOCATION
    # UNGROUPED SKILL PERCENTAGES PER LEVEL

    # group london together
    skill_sample_df["itl_2_name"] = np.where(
        skill_sample_df["itl_2_name"].str.contains("London"),
        "London",
        skill_sample_df["itl_2_name"],
    )

    # group scotland together
    skill_sample_df["itl_2_name"] = np.where(
        skill_sample_df["itl_2_name"].str.contains("Scotland"),
        "Scotland",
        skill_sample_df["itl_2_name"],
    )

    skill_sample_df["itl_2_name"] = np.where(
        skill_sample_df["itl_2_name"].str.contains("Highlands"),
        "Scotland",
        skill_sample_df["itl_2_name"],
    )

    # =================================

    skill_levels_list = list(job_id_2_skill_hier_mentions_per_lev.values())
    job_id_len = skill_sample_df.job_id.nunique()

    skill_sums = defaultdict(list)
    for d in skill_levels_list:
        for k, v in d.items():
            skill_sums[k].append(list(v))

    percentage_skills_by_skill_level = {str(_): list() for _ in range(5)}
    for group_num in ["0", "1", "2", "3", "4"]:
        skill_sum = Counter()
        for skill_dict_list in skill_sums[group_num]:
            skill_sum.update(skill_dict_list)
        percentage_skills_by_skill_level[group_num].append(
            {esco_code2name.get(k): v / job_id_len for k, v in skill_sum.items()}
        )

    # =================================

    sector_2_job_ids, percentage_skill_locs_df = find_skill_proportions_per_group(
        skill_sample_df, job_id_2_skill_count, skill_id_2_ix, "itl_2_name"
    )

    top_skills_per_location = get_top_skills_per_group(
        percentage_skill_locs_df, esco_code2name, top_n=None
    )

    top_skills_per_location_no_trans = get_top_skills_per_group(
        percentage_skill_locs_df,
        esco_code2name,
        top_n=None,
        esco_id_2_trans_flag=esco_id_2_trans_flag,
    )

    top_transversal_skills_per_location = get_only_top_transversal_skills_per_group(
        percentage_skill_locs_df,
        esco_code2name,
        top_n=None,
        esco_id_2_trans_flag=esco_id_2_trans_flag,
    )

    number_job_adverts_per_location = (
        skill_sample_df.groupby("itl_2_name")["job_id"].count().to_dict()
    )

    percentage_skill_by_group_list = []
    for group_num in ["0", "1", "2", "3", "4"]:
        _, percentage_skill_by_group = find_skill_proportions_per_group(
            skill_sample_df,
            {
                job_id: skills[group_num]
                for job_id, skills in job_id_2_skill_hier_mentions_per_lev.items()
            },
            skill_id_2_ix,
            skill_counts=False,
            group="itl_2_name",
        )
        top_skill_by_group_per_group = get_top_skills_per_group(
            percentage_skill_by_group, esco_code2name, top_n=None
        )
        percentage_skill_by_group_list.append(top_skill_by_group_per_group)

    # ==============================================================================

    percentage_skill_by_group_list = []
    percentage_skill_by_group_list_no_trans = []
    percentage_trans_skill_by_group = []
    for group_num in ["0", "1", "2", "3", "4"]:
        _, percentage_skill_by_group = find_skill_proportions_per_group(
            skill_sample_df,
            {
                job_id: skills[group_num]
                for job_id, skills in job_id_2_skill_hier_mentions_per_lev.items()
            },
            skill_id_2_ix,
            skill_counts=False,
            group="itl_2_name",
        )
        top_skill_by_group_per_group = get_top_skills_per_group(
            percentage_skill_by_group, esco_code2name, top_n=None
        )
        top_skill_by_group_per_sector_no_trans = get_top_skills_per_group(
            percentage_skill_by_group,
            esco_code2name,
            top_n=None,
            esco_id_2_trans_flag=esco_id_2_trans_flag,
        )
        top_trans_skills_by_group = get_only_top_transversal_skills_per_group(
            percentage_skill_by_group,
            esco_code2name,
            top_n=None,
            esco_id_2_trans_flag=esco_id_2_trans_flag,
        )

        percentage_skill_by_group_list.append(top_skill_by_group_per_group)
        percentage_skill_by_group_list_no_trans.append(
            top_skill_by_group_per_sector_no_trans
        )
        percentage_trans_skill_by_group.append(top_trans_skills_by_group)
    # ==============================================================================

    all_location_data = {}
    for location_name, num_ads in number_job_adverts_per_location.items():
        all_location_data[location_name] = {
            "num_ads": num_ads,
            "top_skills": {
                "all": top_skills_per_location[location_name],
                "0": percentage_skill_by_group_list[0][location_name],
                "1": percentage_skill_by_group_list[1][location_name],
                "2": percentage_skill_by_group_list[2][location_name],
                "3": percentage_skill_by_group_list[3][location_name],
                "4": percentage_skill_by_group_list[4][location_name],
            },
            "top_skills_no_transversal": {
                "all": top_skills_per_location_no_trans[location_name],
                "0": percentage_skill_by_group_list_no_trans[0][location_name],
                "1": percentage_skill_by_group_list_no_trans[1][location_name],
                "2": percentage_skill_by_group_list_no_trans[2][location_name],
                "3": percentage_skill_by_group_list_no_trans[3][location_name],
                "4": percentage_skill_by_group_list_no_trans[4][location_name],
            },
            "top_transversal_skills": {
                "all": top_transversal_skills_per_location[location_name],
                "2": percentage_trans_skill_by_group[2][location_name],
                "4": percentage_trans_skill_by_group[4][location_name],
            },
        }
    # final dfs to save
    loc_dfs = []
    for loc, skill_info in all_location_data.items():
        loc_df = pd.DataFrame(skill_info["top_skills"]["3"], index=["skill_percent"]).T
        loc_df["region"] = loc
        loc_df = (
            loc_df.reset_index()
            .rename(columns={"index": "skill"})
            .sort_values("skill")
            .reset_index(drop=True)
        )
        loc_dfs.append(loc_df)
    all_loc_df = pd.concat(loc_dfs)

    all_loc_df["total_skill_percentage"] = all_loc_df.skill.map(
        percentage_skills_by_skill_level["3"][0]
    )

    all_loc_df["location_quotident"] = (
        all_loc_df["skill_percent"] / all_loc_df["total_skill_percentage"]
    )
    all_loc_df["location_difference"] = (
        all_loc_df["skill_percent"] - all_loc_df["total_skill_percentage"]
    )
    all_loc_df = all_loc_df.drop(columns="total_skill_percentage")

    all_loc_df["location_change"] = all_loc_df["location_quotident"] - 1

    all_loc_df["absolute_location_change"] = all_loc_df.location_change.abs()

    # get number of job ads per location from dictionary
    num_job_ads_per_loc = {
        loc: loc_info["num_ads"] for loc, loc_info in all_location_data.items()
    }
    all_loc_df["num_ads"] = all_loc_df.region.map(num_job_ads_per_loc)
    all_loc_df["num_ads_per_skill"] = all_loc_df.num_ads * all_loc_df.skill_percent

    # make sure there are at least 100 job adverts associated to a given skill and there are at least 500 job adverts
    # associated to a given location
    all_loc_df = all_loc_df.query("num_ads_per_skill > 100").query("num_ads > 500")

    save_to_s3(
        s3,
        bucket_name,
        all_location_data,
        os.path.join(
            s3_folder,
            "streamlit_viz",
            "top_skills_per_loc_sample.json",
        ),
    )

    save_to_s3(
        s3,
        bucket_name,
        all_loc_df.reset_index(drop=True),
        os.path.join(
            s3_folder,
            "streamlit_viz",
            "top_skills_per_loc_quotident_sample.csv",
        ),
    )
