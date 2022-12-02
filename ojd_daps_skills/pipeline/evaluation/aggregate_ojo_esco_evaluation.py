"""Script to compare top skills from ESCO occupations and from 
OJO occupations

python ojd_daps_skills/pipeline/evaluation/aggregate_ojo_esco_evaluation.py
"""
from ojd_daps_skills import PROJECT_DIR, config, bucket_name
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)
import pandas as pd
import itertools

s3 = get_s3_resource()


def get_job_adverts(esco_job_titles: list, ojo_job_count: int) -> pd.DataFrame:
    """Gets sample of deduplicated job ads,
    job titles and skills where the job title is in the ESCO occupations data AND the number
    of job adverts associated to the job title is over ojo_job_count.

    Args:
        esco_job_titles (list): list of all possible cleaned ESCO occupations.
        ojo_job_count: number of job adverts per occupation threshold.

    Returns:
        pd.DataFrame: A dataframe of occupations and skills per occupation in OJO
                     that are also in ESCO AND have at least 100 job adverts associated
                     to the occupation.
    """
    deduped_skills_sample = load_s3_data(
        s3,
        bucket_name,
        "escoe_extension/outputs/data/model_application_data/dedupe_analysis_skills_sample_temp_fix.json",
    )
    deduped_skills_sample_df = pd.DataFrame(deduped_skills_sample)[
        ["job_id", "occupation", "SKILL"]
    ].dropna()
    deduped_skills_sample_df["clean_skills"] = deduped_skills_sample_df["SKILL"].apply(
        lambda skills: list(set([skill[1][0] for skill in skills]))
    )
    deduped_skills_sample_df[
        "occupation"
    ] = deduped_skills_sample_df.occupation.str.lower()

    ##only return OJO jobs that are in ESCO and that have over 100 occurances
    deduped_skills_sample_df = deduped_skills_sample_df[
        deduped_skills_sample_df.occupation.isin(esco_job_titles)
    ]
    deduped_skills_sample_df = deduped_skills_sample_df.groupby("occupation").filter(
        lambda j: len(j) >= ojo_job_count
    )

    return deduped_skills_sample_df


def get_esco_data(esco_data_dir) -> pd.DataFrame:
    """Loads and merges ESCO csv files from s3. Returns merged DataFarme and
    ESCO's list of transversal skills.

    Inputs:
        esco_data (list): list of paths to local ESCO data files.

    Outputs:
        esco_skills_dict (pd.DataFrame): Merged DataFrame of ESCO jobs with associated alternative
        job titles and skills.
    """

    esco_occs = load_s3_data(
        s3, bucket_name, "escoe_extension/inputs/data/esco/occupations_en.csv"
    )
    esco_occ_skills_walk = load_s3_data(
        s3, bucket_name, "escoe_extension/inputs/data/esco/occupationSkillRelations.csv"
    )
    esco_skills = load_s3_data(
        s3, bucket_name, "escoe_extension/inputs/data/esco/skills_en.csv"
    )

    esco_occ_skills_walk_merged = pd.merge(
        esco_occs, esco_occ_skills_walk, left_on="conceptUri", right_on="occupationUri"
    )
    esco_occ_skills_merged = pd.merge(
        esco_occ_skills_walk_merged,
        esco_skills,
        left_on="skillUri",
        right_on="conceptUri",
    )

    esco_occ_skills_merged = esco_occ_skills_merged[
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
        esco_occ_skills_merged[alt_col] = esco_occ_skills_merged[alt_col].str.split(
            "\n"
        )

    return esco_occ_skills_merged.dropna()


if __name__ == "__main__":
    ojo_job_count = config["ojo_job_count"]
    esco_data_path = config["esco_path"]
    output_path = config["evaluation_results_v1_path"]
    esco_data_dir = config["esco_data_dir"]

    # load data
    ##ESCO data
    esco_jobs = get_esco_data(esco_data_path)
    esco_jobs["all_esco_job_titles"] = esco_jobs.apply(
        lambda j: j["alt_esco_job_titles"] + [j["esco_job_title"]], axis=1
    )

    all_esco_job_titles = [
        job
        for job in list(
            set(list(itertools.chain(*esco_jobs.all_esco_job_titles.to_list())))
        )
    ]

    ojo_job_adverts = get_job_adverts(all_esco_job_titles, ojo_job_count)

    ## Compare ESCO and OJO skills
    # get skill percents per occupation
    deduped_skills_sample_df_exploded = ojo_job_adverts.explode("clean_skills")

    ## Compare ESCO and OJO skills
    skill_percent_occ = (
        deduped_skills_sample_df_exploded.groupby(
            "occupation"
        ).clean_skills.value_counts()
        / deduped_skills_sample_df_exploded.groupby("occupation").clean_skills.nunique()
    )
    skill_percent_occ = (
        pd.DataFrame(skill_percent_occ)
        .rename(columns={"clean_skills": "skill_percent"})
        .reset_index()
    )

    # generate skill threshold based on distribution of skill percentages
    skill_thresholds = skill_percent_occ.groupby("occupation")[
        "skill_percent"
    ].describe()["75%"]
    skill_percent_occ["skill_percent_threshold"] = skill_percent_occ["occupation"].map(
        skill_thresholds
    )

    # compare OJO and ESCO skills
    ojo_esco_dict = dict()
    for occupation, occ_data in skill_percent_occ.groupby("occupation"):
        esco_skills = set(
            esco_jobs[
                esco_jobs["all_esco_job_titles"].apply(lambda x: occupation in x)
            ]["esco_skill"]
        )
        per_thresh = occ_data["skill_percent_threshold"].iloc[0]
        ojo_skill = set(
            skill_percent_occ[skill_percent_occ["skill_percent"] > per_thresh][
                "clean_skills"
            ].tolist()
        )
        in_both_ojo_esco, in_ojo_not_esco, in_esco_not_ojo = (
            set.intersection(esco_skills, ojo_skill),
            list(ojo_skill - esco_skills),
            list(esco_skills - ojo_skill),
        )
        if ojo_skill:
            ojo_esco_dict[occupation] = {
                "no_of_job_adverts": ojo_job_adverts[
                    ojo_job_adverts.occupation == occupation
                ]["job_id"].nunique(),
                "in_both_ojo_esco": list(in_both_ojo_esco),
                "in_ojo_not_esco": in_ojo_not_esco,
                "in_esco_not_ojo": in_esco_not_ojo,
                "skills_in_ojo_esco_percent": len(in_both_ojo_esco)
                / len(esco_skills)
                * 100,
            }
    # Save occupation-level accuracy results to s3
    save_to_s3(
        s3,
        bucket_name,
        ojo_esco_dict,
        "escoe_extension/outputs/evaluation/aggregate_ojo_esco/ojo_esco_occupation_skills_results_v2.json",
    )
