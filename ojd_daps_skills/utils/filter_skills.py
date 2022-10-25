"""
One off script to filter job skills sample using deduplicated job ids
"""
from ojd_daps_skills import bucket_name
from ojd_daps_skills.getters.data_getters import (
    load_s3_data,
    get_s3_resource,
    save_to_s3,
)

from tqdm import tqdm
import itertools

if __name__ == "__main__":

    s3 = get_s3_resource()

    job_skills = load_s3_data(
        s3,
        bucket_name,
        "escoe_extension/outputs/data/model_application_data/job_ad_to_skills_sample.json",
    )
    job_ads_deduped = load_s3_data(
        s3,
        bucket_name,
        "escoe_extension/outputs/data/model_application_data/deduplicated_job_ids_6_weeks.csv",
    )

    def list_chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    # initially filter a bit based on first 4 digits of job ids in job_skills
    job_skills_starts = list(set([x["job_id"][:4] for x in job_skills]))
    job_ads_deduped_ids = [str(x) for x in list(job_ads_deduped.job_id)]
    job_ads_deduped_ids_starts = list(
        itertools.chain(
            *[
                [x for x in job_ads_deduped_ids if x.startswith(start)]
                for start in job_skills_starts
            ]
        )
    )

    # filter in chunks
    all_job_skills_deduped = []
    for batch_ids in tqdm(list_chunks(job_ads_deduped_ids_starts, 5000)):
        job_skills_deduped = list(
            filter(lambda x: x["job_id"] in batch_ids, job_skills)
        )
        all_job_skills_deduped.extend(job_skills_deduped)

    save_to_s3(
        s3,
        bucket_name,
        all_job_skills_deduped,
        "escoe_extension/outputs/data/model_application_data/deduplicated_job_ad_to_skills_sample_6_weeks.json",
    )
