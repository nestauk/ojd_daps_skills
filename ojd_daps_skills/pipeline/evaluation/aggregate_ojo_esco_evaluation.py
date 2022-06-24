"""Script to evaluate the first iteration of the skills algorithm


"""
##############################################################
from ojd_daps_skills.utils.sql_conn import est_conn
from ojd_daps_skills import PROJECT_DIR, config, bucket_name
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    get_s3_data_paths,
    load_s3_data,
    save_to_s3,
)
import glob
import os
import string
import pandas as pd
import itertools
from datetime import datetime as date

##############################################################

s3 = get_s3_resource()


def clean_job_title(job_title):
    """Cleans job title to lowercase, remove punctuation, numbers and "bad" words."""

    job_title = job_title.lower().translate(
        str.maketrans("", "", string.punctuation + string.digits)
    )
    job_title = " ".join(job_title.split())
    job_title = " ".join([job for job in job_title.split() if job not in ["amp"]])

    return job_title.strip()


def get_job_adverts(conn, esco_job_titles: list, ojo_job_count: int) -> pd.DataFrame:
    """Queries SQL db to return merged dataframe of job ids, job skills and
    job titles where the job title is in the ESCO occupations data AND the number
    of job adverts associated to the job title is over ojo_job_count.

    Args:
        conn: Engine to Nesta's SQL database.
        esco_job_titles (list): list of all possible cleaned ESCO occupations.
        ojo_job_count: number of job adverts per occupation threshold.

    Returns:
        pd.DataFrame: A dataframe of occupations and skills per occupation in OJO
                     that are also in ESCO AND have at least 100 job adverts associated
                     to the occupation.
    """
    query_job_titles = "SELECT id, job_title_raw" " FROM raw_job_adverts "
    job_titles = pd.read_sql(query_job_titles, conn)

    clean_jobs = dict(
        zip(
            list(set(job_titles.job_title_raw)),
            [clean_job_title(job) for job in list(set(job_titles.job_title_raw))],
        )
    )

    job_titles["clean_ojo_job_title"] = job_titles.job_title_raw.map(clean_jobs)

    ##only return OJO jobs that are in ESCO and that have over 100 occurances
    job_titles = job_titles[job_titles.clean_ojo_job_title.isin(esco_job_titles)]
    job_titles = job_titles.groupby("clean_ojo_job_title").filter(
        lambda j: len(j) > ojo_job_count
    )

    # query job skills based on job titles with over n occurances in ESCO
    query_job_skills = f"SELECT job_id, preferred_label FROM job_ad_skill_links WHERE job_id IN {tuple(set(job_titles.id))}"
    job_skills = pd.read_sql(query_job_skills, conn)

    return pd.merge(job_skills, job_titles, left_on="job_id", right_on="id")


def get_esco_data(esco_data_dir) -> pd.DataFrame:
    """Loads and merges ESCO csv files from s3. Returns merged DataFarme and
    ESCO's list of transversal skills.
    
    Inputs:
        esco_data (list): list of paths to local ESCO data files.

    Outputs:
        esco_skills_dict (pd.DataFrame): Merged DataFrame of ESCO jobs with associated alternative
        job titles and skills.
    """
    esco_paths = get_s3_data_paths(s3, bucket_name, esco_data_dir, "*.csv")
    esco_dfs = [load_s3_data(s3, bucket_name, esco_path) for esco_path in esco_paths]

    esco_occ_skills = pd.merge(
        pd.merge(
            esco_dfs[0], esco_dfs[1], left_on="occupationUri", right_on="conceptUri"
        ),
        esco_dfs[2],
        left_on="skillUri",
        right_on="conceptUri",
    )
    esco_occ_skills = esco_occ_skills[
        ["preferredLabel_x", "altLabels_x", "preferredLabel_y", "altLabels_y"]
    ].rename(
        columns={
            "preferredLabel_x": "esco_job_title",
            "altLabels_x": "alt_esco_job_titles",
            "preferredLabel_y": "esco_skill",
            "altLabels_y": "alt_esco_skills",
        }
    )

    for alt_col in "alt_esco_skills", "alt_esco_job_titles":
        esco_occ_skills[alt_col] = esco_occ_skills[alt_col].str.split("\n")

    return esco_occ_skills


if __name__ == "__main__":
    ojo_job_count = config["ojo_job_count"]
    esco_data_path = config["esco_path"]
    output_path = config["evaluation_results_path"]
    esco_data_dir = config["esco_data_dir"]

    # load data
    ##ESCO data
    esco_jobs = get_esco_data(esco_data_path)
    esco_jobs["all_esco_job_titles"] = esco_jobs.apply(
        lambda j: j["alt_esco_job_titles"] + [j["esco_job_title"]], axis=1
    )
    all_esco_job_titles = [
        clean_job_title(job)
        for job in list(
            set(list(itertools.chain(*esco_jobs.all_esco_job_titles.to_list())))
        )
    ]

    ##OJO jobs in ESCO
    conn = est_conn()
    ojo_job_adverts = get_job_adverts(conn, all_esco_job_titles, ojo_job_count)

    ## Compare ESCO and OJO skills
    # get skill percents per occupation
    skill_percent_occ = (
        ojo_job_adverts.groupby(["clean_ojo_job_title", "preferred_label"])[
            "job_id"
        ].nunique()
        / ojo_job_adverts.groupby("clean_ojo_job_title")["job_id"].nunique()
        * 100
    )
    skill_percent_occ = skill_percent_occ.reset_index().rename(
        columns={"job_id": "skill_percent"}
    )

    # generate skill threshold based on distribution of skill percentages
    skill_thresholds = (
        skill_percent_occ.groupby("clean_ojo_job_title")["skill_percent"].describe()[
            "50%"
        ]
        + 0.5
        * skill_percent_occ.groupby("clean_ojo_job_title")["skill_percent"].describe()[
            "std"
        ]
    )
    skill_percent_occ["skill_percent_threshold"] = skill_percent_occ[
        "clean_ojo_job_title"
    ].map(skill_thresholds)

    # compare OJO and ESCO skills
    ojo_esco_dict = dict()
    for occupation, occ_data in skill_percent_occ.groupby("clean_ojo_job_title"):
        esco_skills = set(
            esco_jobs[
                esco_jobs["all_esco_job_titles"].apply(lambda x: occupation in x)
            ]["esco_skill"]
        )
        skills_above_threshold = [
            i
            for i, skill in enumerate(occ_data["skill_percent"])
            if skill > occ_data["skill_percent_threshold"].iloc[0]
        ]
        ojo_skill = set(
            list(occ_data["preferred_label"])[i] for i in skills_above_threshold
        )
        in_both_ojo_esco, in_ojo_not_esco, in_esco_not_ojo = (
            set.intersection(esco_skills, ojo_skill),
            list(ojo_skill - esco_skills),
            list(esco_skills - ojo_skill),
        )
        if ojo_skill:
            ojo_esco_dict[occupation] = {
                "in_both_ojo_esco": list(in_both_ojo_esco),
                "in_ojo_not_esco": in_ojo_not_esco,
                "in_esco_not_ojo": in_esco_not_ojo,
                "skills_in_ojo_esco_percent": len(in_both_ojo_esco) / len(ojo_skill),
            }

    # Save occupation-level accuracy results to s3
    save_to_s3(s3, bucket_name, ojo_esco_dict, output_path)
