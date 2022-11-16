"""Script to extract OJO V2 ESCO and Lightcast skills to sample of job adverts with OJO V1 and Lighcast skill extracted skills."""
from ojd_daps_skills.getters.data_getters import (
    load_file,
    load_s3_data,
    get_s3_resource,
    save_to_s3,
)
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills
import pandas as pd
from datetime import datetime as date

from ojd_daps_skills import bucket_name


def extract_skills_v2(config_name: str, ojo_emsi_skills_v1):
    """uses skill extractor to extract and map skills onto a given taxonomy."""

    taxonomy_name = config_name.split("_")[-1]

    es = ExtractSkills(config_name=config_name, local=False)
    es.load()
    ojo_v2_esco_skills = es.extract_skills(job_adverts, map_to_tax=True)

    for job_info, extracted_skills in zip(
        ojo_emsi_skills_v1.items(), ojo_v2_esco_skills
    ):
        if "SKILL" in extracted_skills.keys():
            job_info[1][f"ojo_v2_{taxonomy_name}_skills"] = [
                skill[1][0] for skill in extracted_skills["SKILL"]
            ]
        else:
            job_info[1][f"ojo_v2_{taxonomy_name}_skills"] = []

    return ojo_emsi_skills_v1


if __name__ == "__main__":

    S3 = get_s3_resource()

    ojo_emsi_skills_v1_path = (
        "escoe_extension/outputs/evaluation/ojo_esmi_skills/ojo_esmi_skills_v1.json"
    )
    ojo_emsi_skills_v1 = load_file(ojo_emsi_skills_v1_path)

    job_adverts = [job_info["job_ad_text"] for job_info in ojo_emsi_skills_v1.values()]

    ojo_emsi_skills_v1_v2_lightcast = extract_skills_v2(
        "extract_skills_lightcast", ojo_emsi_skills_v1
    )
    ojo_emsi_skills_v1_v2_esco = extract_skills_v2(
        "extract_skills_esco", ojo_emsi_skills_v1_v2_lightcast
    )

    skills_extractor_df = pd.DataFrame(ojo_emsi_skills_v1_v2_esco).T

    date_stamp = str(date.today().date()).replace("-", "")
    save_to_s3(
        S3,
        bucket_name,
        skills_extractor_df,
        f"escoe_extension/outputs/evaluation/{date_stamp}_extracted_skills_all_extractors.csv",
    )
