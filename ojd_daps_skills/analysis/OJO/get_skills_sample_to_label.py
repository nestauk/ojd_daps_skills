"""
A sample of the skills to label for quality
"""

import os
from datetime import date
import random

import pandas as pd
import numpy as np
import altair as alt

from ojd_daps_skills.utils.plotting import NESTA_COLOURS, nestafont, configure_plots
from ojd_daps_skills.utils.save_plotting import AltairSaver

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    get_s3_data_paths,
    load_s3_json,
    load_s3_data,
    save_to_s3,
)
from ojd_daps_skills import bucket_name

s3 = get_s3_resource()

s3_folder = "escoe_extension/outputs/data/model_application_data"

# The skill sample
file_name = os.path.join(s3_folder, "dedupe_analysis_skills_sample.json")
skill_sample = load_s3_data(s3, bucket_name, file_name)

job_2_skill = []
for job_adverts in skill_sample:
    if job_adverts["SKILL"]:
        for skill in job_adverts["SKILL"]:
            job_2_skill.append({"job_id": job_adverts["job_id"], "skill": skill})

random.seed(42)
random_job_2_skill = random.choices(job_2_skill, k=200)

random_job_2_skill_df = pd.DataFrame(random_job_2_skill)

random_job_2_skill_df.to_csv("skills_sample_to_tag.csv")
