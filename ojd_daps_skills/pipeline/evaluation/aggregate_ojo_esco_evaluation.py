"""Script to evaluate the aggregated overlap between extracted skills
from OJO and ESCO skills per occupation.
"""
###########################################
from ojd_daps_skills.utils.sql_conn import est_conn
from ojd_daps_skills import PROJECT_DIR, config
from ojd_daps_skills.getters.data_getters import load_local_data, save_json_dict
import glob
import os
import string
import pandas as pd
import itertools

###########################################


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


def get_esco_data(esco_data: list) -> list:
    """Loads, cleans and merges local ESCO csv files. Converts dataframe to tuple
    where key is esco_job_title and values include all skills, including alternative
    skills and alternative job titles.

    Inputs:
        esco_data (list): list of paths to local ESCO data files.

    Outputs:
        esco_skills_dict (dict): Dictionary of esco jobs with associated alternative
        job titles and skills.
    """
    esco_dfs = [load_local_data(esco_path) for esco_path in esco_data_path]
    esco_dfs[1].rename(columns={"conceptUri": "skillUri"}, inplace=True), esco_dfs[
        2
    ].rename(columns={"conceptUri": "occupationUri"}, inplace=True)
    esco_occ_skills = pd.merge(
        pd.merge(esco_dfs[0], esco_dfs[1], on="skillUri"),
        esco_dfs[2],
        on="occupationUri",
    )
    esco_occ_skills = esco_occ_skills[
        ["preferredLabel_x", "altLabels_x", "preferredLabel_y", "altLabels_y"]
    ].rename(
        columns={
            "preferredLabel_x": "esco_skill",
            "altLabels_x": "alt_esco_skills",
            "preferredLabel_y": "esco_job_title",
            "altLabels_y": "alt_esco_job_titles",
        }
    )

    for alt_col in "alt_esco_skills", "alt_esco_job_titles":
        esco_occ_skills[alt_col] = esco_occ_skills[alt_col].str.split("\n")

    esco_occ_skills["esco_skill"] = esco_occ_skills["esco_skill"].apply(lambda x: [x])
    esco_occ_skills = [
        tuple(j) for j in esco_occ_skills.groupby("esco_job_title").sum().to_numpy()
    ]

    return esco_occ_skills


if __name__ == "__main__":
    ojo_job_count = config["ojo_job_count"]
    esco_data_path = glob.glob(
        os.path.join(str(PROJECT_DIR / config["esco_path"]), "*.csv")
    )

    output_path = str(PROJECT_DIR) + config["evaluation_results_path"]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load data
    ##ESCO data
    esco_jobs = get_esco_data(esco_data_path)
    all_esco_job_titles = list(itertools.chain(*[esco[2] for esco in esco_jobs]))

    ##OJO jobs in ESCO
    conn = est_conn()
    ojo_job_adverts = get_job_adverts(conn, all_esco_job_titles, ojo_job_count)

    # Compare OJO and ESCO skills per occupation
    ojo_esco_dict = dict()
    for ojo_job_title, ojo_data in ojo_job_adverts.groupby("clean_ojo_job_title"):

        essential_esco_skills = [
            esco_job[0] for esco_job in esco_jobs if ojo_job_title in esco_job[2]
        ]
        alt_esco_skills = [
            esco_job[1] for esco_job in esco_jobs if ojo_job_title in esco_job[2]
        ]

        if essential_esco_skills != []:
            ojo_esco_essen_skills = list(
                set(ojo_data["preferred_label"]) & set(essential_esco_skills[0])
            )

        if alt_esco_skills != []:
            ojo_esco_alt_skills = list(
                set(ojo_data["preferred_label"]) & set(alt_esco_skills[0])
            )

            ojo_esco_dict[ojo_job_title] = {
                "no_of_adverts": len(ojo_data["job_id"]),
                "ojo_esco_essen_skills": ojo_esco_essen_skills,
                "ojo_esco_alt_skills": ojo_esco_alt_skills,
                "essen_accuracy": len(ojo_esco_essen_skills)
                / len(essential_esco_skills[0]),
                "alt_accuracy": len(ojo_esco_alt_skills) / len(alt_esco_skills[0]),
            }

    # Save occupation-level accuracy results to file
    save_json_dict(
        ojo_esco_dict,
        os.path.join(output_path, f"ojo_esco_occupation_skills_results.json"),
    )
