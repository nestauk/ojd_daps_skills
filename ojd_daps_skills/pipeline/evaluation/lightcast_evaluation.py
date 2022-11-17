"""
Script to extract lightcast skills from a random sample of 50 OJO job ads.

To run the script, python ojd_daps_skills/pipeline/evaluation/lightcast_evaluation.py --client-id CLIENT_ID --client-secret CLIENT_SECRET

You will need to on the Nesta wifi or turn on the Nesta VPN. 
"""
import random
import requests
import json
from argparse import ArgumentParser
import time
import pandas as pd

from ojd_daps_skills import config, bucket_name, logger

from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills


from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)
from datetime import datetime as date


def get_job_advert_skills(conn, job_ids: list) -> dict:
    """Queries SQL db to return dataframe of job ids and skills.
    Args:
        conn: Engine to Nesta's SQL database.
        job_ids (list): list of job_ids to get skills for.

    Returns:
        dict: A dictionary of OJO skills associated to each job id.
    """
    query_job_skills = f"SELECT job_id, preferred_label FROM job_ad_skill_links WHERE job_id IN {tuple(set(job_ids))}"
    job_skills = pd.read_sql(query_job_skills, conn)

    return job_skills.groupby("job_id")["preferred_label"].apply(list).to_dict()


def get_lightcast_access_token(client_id: str, client_secret: str) -> str:
    """Generates temporary access token needed to query lightcast skills API.
    
    Inputs:
        client_id (str): Client ID from generated lightcast skills API credentials.
        client_secret (str): Client secret from generated lightcast skills API credentials.
    
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

def get_extracted_lightcast_skill(extracted_lightcast_skill):
    """Helper function to return list of mapped lightcast skills"""
    if 'SKILL' in extracted_lightcast_skill.keys():
        return list(set([skill[1][0] for skill in extracted_lightcast_skill['SKILL']]))
    else:
        return extracted_lightcast_skill


def extract_lightcast_skills(
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

    time.sleep(12)  # make 1 API call every 12 seconds
    if response.ok:
        return response.json()["data"]
    else:
        return response


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--client-id", help="lightcast skills API client id",
    )

    parser.add_argument("--client-secret", help="lightcast skills API client secret.")

    args = parser.parse_args()

    client_id = args.client_id
    client_secret = args.client_secret

    # load sample
    ojo_job_ads_sample = load_s3_data(
        get_s3_resource(), bucket_name, config["ojo_random_sample_path"]
    )

    # then random sample 50 job adverts
    random.seed(72)
    random_job_ids = random.sample(list(ojo_job_ads_sample.keys()), 50)
    ojo_job_ads_sample_50 = {k:v['description'] for k, v in ojo_job_ads_sample.items() if k in random_job_ids}

    logger.info("randomly sampled 50 job adverts")

    #extract our lightcast skills
    #get v2 skills
    skills_extractor = ExtractSkills(config_name="extract_skills_lightcast_evaluation", s3=True)

    skills_extractor.load()

    extracted_skills = skills_extractor.extract_skills([advert.replace('[', '').replace(']', '').strip() for advert in list(ojo_job_ads_sample_50.values())])
    
    # get lightcast access code
    access_code = get_lightcast_access_token(client_id, client_secret)

    logger.info("got lightcast access token")

    # get extracted lightcast skills
    ojo_lightcast_skills = dict()
    for i, (job_id, ojo_job_ad_data) in enumerate(ojo_job_ads_sample_50.items()):
        lightcast_skills = extract_lightcast_skills(
            access_code,
            ojo_job_ad_data,
            config["esmi_confidence_threshold"],
        )
        if not isinstance(lightcast_skills, requests.models.Response):  # if its not an error
            ojo_lightcast_skills[job_id] = {
                "job_ad_text": ojo_job_ad_data,
                "ojo_skills": get_extracted_lightcast_skill(extracted_skills[i]),
                "lightcast_skills": [skill["skill"]["name"] for skill in lightcast_skills]}

    logger.info("extracted lightcast skills")

    save_to_s3(
        get_s3_resource(),
        bucket_name,
        ojo_lightcast_skills,
        ("escoe_extension/outputs/evaluation/ojo_emsi_skills/ojo_lightcast_skills_" + f"{str(date.today().date()).replace('-', '')}.json")
    )
