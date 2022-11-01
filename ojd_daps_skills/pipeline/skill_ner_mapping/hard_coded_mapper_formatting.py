"""One of script to format manually labelled skill spans
in format for prev_skills_lookup_sample.json
"""
from ojd_daps_skills.utils.text_cleaning import short_hash
from ojd_daps_skills import bucket_name
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    save_to_s3,
    load_s3_data,
)

import pandas as pd

if __name__ == "__main__":

    s3 = get_s3_resource()

    hard_coded_skills = (
        load_s3_data(
            s3,
            bucket_name,
            "escoe_extension/inputs/data/skill_mappings/hard_labelled_skills.csv",
        )
        .query("~new_label.isna()")
        .assign(ojo_job_skill_hash=lambda df: df.extracted_skill.apply(short_hash))
        .assign(hash_index=lambda df: df.ojo_job_skill_hash)
        .rename(
            columns={
                "extracted_skill": "ojo_skill",
                "esco_code": "match_id",
                "new_label": "match_skill",
            }
        )[
            [
                "ojo_skill",
                "ojo_job_skill_hash",
                "match_skill",
                "match_id",
                "hash_index",
            ]
        ]
    )

    hard_coded_skills_dict = hard_coded_skills.set_index("hash_index").T.to_dict()

    save_to_s3(
        s3,
        bucket_name,
        hard_coded_skills_dict,
        "escoe_extension/outputs/data/skill_ner_mapping/hardcoded_ojo_esco_lookup.json",
    )
