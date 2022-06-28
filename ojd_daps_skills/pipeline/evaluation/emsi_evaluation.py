"""
Script to extract EMSI skills from a random sample of 50 OJO job ads.

To run the script, python emsi_evaluation.py --client-id CLIENT_ID --client-secret CLIENT_SECRET
"""
import random
import requests
import json
from argparse import ArgumentParser
import time
import pandas as pd

from ojd_daps_skills import config, bucket_name

from ojd_daps_skills.utils.sql_conn import est_conn

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)


def get_job_advert_skills(conn, job_ids: list) -> pd.DataFrame:
    """Queries SQL db to return dataframe of job ids and skills.
    Args:
        conn: Engine to Nesta's SQL database.
        job_ids (list): list of job_ids to get skills for.

    Returns:
        pd.DataFrame: A dataframe of skills associated to each job id.
    """

    query_job_skills = f"SELECT job_id, preferred_label FROM job_ad_skill_links WHERE job_id IN {tuple(set(job_ids))}"
    job_skills = pd.read_sql(query_job_skills, conn)

    return job_skills.groupby("job_id")["preferred_label"].apply(list).to_dict()


def get_emsi_access_token(client_id: str, client_secret: str) -> str:
    """Generates temporary access token needed to query EMSI skills API.
    
    Inputs:
        client_id (str): Client ID from generated EMSI skills API credentials.
        client_secret (str): Client secret from generated EMSI skills API credentials.
    
    Outputs:
        access_token (str): Access token string valid for 1 hour.    
    """

    url = "https://auth.emsicloud.com/connect/token"

    payload = f"client_id={client_id}&client_secret={client_secret}&grant_type=client_credentials&scope=emsi_open"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.request("POST", url, data=payload, headers=headers)

    if response.ok:
        return response.json()["access_token"]
    else:
        return response


def extract_esmi_skills(
    access_token: str, job_advert_text: str, confidence_threshold: int
):
    """Extracts ESMI skills for a given english job advert.
    
    Inputs:
        access_token (str): access token string to pass in requests header.
        job_advert_text (str): Text of job advert.
        confidence_threshold (int): confidence threshold of extracted ESMI skill.
        
    Outputs:
        response (dict): ESMI extracted skills from job advert.
    """
    url = "https://emsiservices.com/skills/versions/latest/extract"

    querystring = {"language": "en"}  # assumes OJO job advert is in english

    payload = {"text": job_advert_text, "confidenceThreshold": confidence_threshold}

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    response = requests.request(
        "POST", url, data=json.dumps(payload), headers=headers, params=querystring
    )

    if response.ok:
        return response.json()["data"]
    else:
        return response


# %%
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--client-id", help="EMSI skills API client id",
    )

    parser.add_argument("--client-secret", help="EMSI skills API client secret.")

    args = parser.parse_args()

    client_id = args.client_id
    client_secret = args.client_secret

    # load sample
    ojo_job_ads_sample = load_s3_data(
        get_s3_resource(), bucket_name, config["ojo_random_sample_path"]
    )

    # get skills
    ##doing it this way round as sometimes the job id is not in the job id skills link db
    conn = est_conn()
    ojo_job_skills = get_job_advert_skills(conn, list(ojo_job_ads_sample.keys()))

    # add skills to dict
    ojo_jobs_with_skills = dict()
    for job_id, ojo_job_ad in ojo_job_ads_sample.items():
        if job_id in list(ojo_job_skills.keys()):
            ojo_jobs_with_skills[job_id] = {
                "job_ad_text": ojo_job_ad["description"],
                "ojo_skills": ojo_job_skills[job_id],
            }

    # then random sample 50 job adverts
    random.seed(72)
    ojo_job_ads_50 = {
        job_id: job_data
        for job_id, job_data in ojo_jobs_with_skills.items()
        if job_id in random.sample(list(ojo_jobs_with_skills.keys()), 50)
    }

    # get EMSI access code
    access_code = get_emsi_access_token(client_id, client_secret)

    # get extracted EMSI skills
    for job_id, ojo_job_ad_data in ojo_job_ads_50.items():
        esmi_skills = extract_esmi_skills(
            access_token,
            ojo_job_ad_data["job_ad_text"],
            config["esmi_confidence_threshold"],
        )
        time.sleep(60)  # make 1 API call a minute
        ojo_job_ad_data[job_id] = {
            "esmi_skills": [skill["skill"]["name"] for skill in esmi_skills["data"]]
        }

    save_to_s3(
        get_s3_resource(), bucket_name, ojo_esmi_skills, config["esmi_ojo_skills_path"]
    )
