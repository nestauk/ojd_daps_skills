"""
Use the ExtractSkills function to extract and map skills for our sample of OJO job adverts
"""
import os
from itertools import islice
from datetime import datetime as date
from collections import Counter

from tqdm import tqdm

from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
    load_json_dict,
)
from ojd_daps_skills import bucket_name, logger

S3 = get_s3_resource()


def chunks(data_dict, chunk_size=100):
    it = iter(data_dict)
    for i in range(0, len(data_dict), chunk_size):
        yield {k: data_dict[k] for k in islice(it, chunk_size)}


def extract_skills_ojo_job_ads(job_adverts, es, train_job_ids):
    """
    Takes raw OJO job adverts, a dict of job id to job advert info
    and outputs a dict of job id to skill extracted and matched
    """
    job_advert_texts_list = []
    job_advert_ix_info = []
    for job_id, job_info in job_adverts.items():
        job_advert_texts_list.append(job_info["description"])
        job_advert_ix_info.append((job_id, True if job_id in train_job_ids else False))

    # Might want to do the next bits in loops since it could take a long time to embed
    # many job adverts
    job_skills_matched = es.extract_skills(job_advert_texts_list, map_to_tax=True)

    if len(job_skills_matched) != len(job_advert_ix_info):
        logger.warning(
            "The number of predictions lists dont match the number of job adverts"
        )

    # Combine the job advert info with the prediction
    ojo_extracted_skills = {}
    for job_ad_info, match_data in zip(job_advert_ix_info, job_skills_matched):
        match_data["in_train?"] = job_ad_info[1]
        ojo_extracted_skills[job_ad_info[0]] = match_data

    return ojo_extracted_skills


if __name__ == "__main__":

    update_lookup_sample = False

    job_adverts_filename = "escoe_extension/inputs/data/skill_ner/data_sample/20220622_sampled_job_ads.json"

    es = ExtractSkills(config_name="extract_skills_esco", local=False)

    if update_lookup_sample:
        es.prev_skill_matches_file_name = ""

    model_train_info = load_json_dict(
        os.path.join(es.config["ner_model_path"], "train_details.json")
    )
    seen_job_ids_dict = model_train_info["seen_job_ids"]
    train_job_ids = set(
        [k for k, v in seen_job_ids_dict.items() if v["train/test"] == "train"]
    )
    job_adverts = load_s3_data(S3, bucket_name, job_adverts_filename)

    es.load()

    batch_size = 500

    ojo_extracted_skills = {}
    for batch_job_adverts in tqdm(chunks(job_adverts, batch_size)):
        ojo_extracted_skills.update(
            extract_skills_ojo_job_ads(batch_job_adverts, es, train_job_ids)
        )

    date_stamp = str(date.today().date()).replace("-", "")
    save_to_s3(
        S3,
        bucket_name,
        ojo_extracted_skills,
        f"escoe_extension/outputs/data/extract_skills/{date_stamp}_ojo_sample_skills_extracted.json",
    )

    if update_lookup_sample:
        # Save out some common skill matches for future use
        skill_mapper = es.skill_mapper

        common_skill_hash = set(
            [
                i
                for i, v in Counter(
                    [h["ojo_job_skill_hash"] for h in es.skill_matches]
                ).most_common(100)
            ]
        )
        sample_skill_hash_to_esco = {}
        for fm in es.skill_matches:
            skill_hash = fm["ojo_job_skill_hash"]
            if skill_hash in common_skill_hash:
                sample_skill_hash_to_esco[skill_hash] = fm

        skill_mapper.save_ojo_esco_mapper(
            "escoe_extension/outputs/data/skill_ner_mapping/ojo_esco_lookup_sample.json",
            sample_skill_hash_to_esco,
        )

    # all 5000 at once:
    #  Took 144.59014081954956 seconds (bert_vectorizer.py:47)
    # didn't wait until the mapping finished but at least 15 mins

    # batch size =10 -> iterations were 9.3, 7.9, 8.17 secs
    # batch size=500 -> 125 secs each
