"""Script to extract OJO V2 ESCO skills to sample of job adverts with Lighcast skill extracted skills.

python ojd_daps_skills/pipeline/evaluation/evaluate_all_skill_extractors.py
"""
from ojd_daps_skills.getters.data_getters import (
    load_s3_data,
    get_s3_resource,
    save_to_s3,
)
from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills
import pandas as pd
from datetime import datetime as date

from ojd_daps_skills import bucket_name

if __name__ == "__main__":

    s3 = get_s3_resource()

    ojo_emsi_skills_path = (
        "escoe_extension/outputs/evaluation/ojo_esmi_skills/ojo_esmi_skills_v1.json"
    )
    ojo_lightcast_skills_path = "escoe_extension/outputs/evaluation/ojo_esmi_skills/ojo_lightcast_skills_20221115.json"

    ojo_emsi_skills = load_s3_data(s3, bucket_name, ojo_emsi_skills_path)
    ojo_lightcast_skills = load_s3_data(s3, bucket_name, ojo_lightcast_skills_path)

    ojo_emsi_skills = (
        pd.DataFrame(ojo_emsi_skills)
        .T.reset_index()[["index", "job_ad_text", "esmi_skills"]]
        .rename(columns={"esmi_skills": "lightcast_skills"})
    )
    ojo_lightcast_skills = pd.DataFrame(ojo_lightcast_skills).T.reset_index()[
        ["index", "job_ad_text", "lightcast_skills"]
    ]

    all_ojo_lightcast_skills = (
        pd.concat([ojo_emsi_skills, ojo_lightcast_skills])
        .rename(columns={"index": "job_id"})
        .drop_duplicates("job_id")
    )

    # esco skills extractor
    skills_extractor = ExtractSkills(config_name="extract_skills_esco")
    skills_extractor.load()
    # extract esco skills
    esco_extracted_skills = skills_extractor.extract_skills(
        [
            ad.replace("[", "").replace("]", "").strip()
            for ad in list(all_ojo_lightcast_skills.job_ad_text)
        ]
    )

    # concatenate lightcast esco skills
    all_ojo_lightcast_skills["ojo_esco_skill"] = esco_extracted_skills
    all_ojo_esco_lightcast_skills = (
        pd.concat(
            [
                all_ojo_lightcast_skills,
                all_ojo_lightcast_skills.ojo_esco_skill.apply(pd.Series),
            ],
            axis=1,
        )
        .drop(["ojo_esco_skill", "EXPERIENCE"], axis=1)
        .rename(columns={"SKILL": "ojo_esco_skill"})
    )

    # save dataframe
    date_stamp = str(date.today().date()).replace("-", "")
    save_to_s3(
        s3,
        bucket_name,
        all_ojo_esco_lightcast_skills,
        f"escoe_extension/outputs/evaluation/{date_stamp}_extracted_skills_all_extractors.csv",
    )
