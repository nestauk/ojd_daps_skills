"""
Use a trained NER model to predict skills and experience spans in a sample of job adverts

Running

python ojd_daps_skills/pipeline/skill_ner/get_skills.py
    --model_path outputs/models/ner_model/20220630/
    --output_file_dir escoe_extension/outputs/data/skill_ner/skill_predictions/
    --job_adverts_filename escoe_extension/inputs/data/skill_ner/data_sample/20220622_sampled_job_ads.json

will make skill predictions on the data in `job_adverts_filename` (an output of `create_data_sample.py`)
using the model loaded from `model_path`. By default this will look for the model on S3,
but if you want to load a locally stored model just add `--use_local_model`.

The output will contain a dictionary of predictions, where each key is the job advert ID,
including a flag for whether this job advert was used in the training of the model or not.


"""
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
    load_json_dict,
)
from ojd_daps_skills import bucket_name
from ojd_daps_skills.pipeline.skill_ner.ner_spacy import JobNER

import spacy
from tqdm import tqdm

from datetime import datetime as date
import json
import os
from argparse import ArgumentParser


def parse_arguments(parser):

    parser.add_argument(
        "--model_path",
        help="The path to the model you want to make predictions with",
        default="outputs/models/ner_model/20220630/",
    )

    parser.add_argument(
        "--output_file_dir",
        help="The S3 folder to output the predictions to",
        default="escoe_extension/outputs/data/skill_ner/skill_predictions/",
    )

    parser.add_argument(
        "--job_adverts_filename",
        help="The S3 path to the job advert dataset you want to make predictions on",
        default="escoe_extension/inputs/data/skill_ner/data_sample/20220622_sampled_job_ads.json",
    )

    parser.add_argument(
        "--use_local_model",
        help="Use the model locally stored",
        action="store_true",
        default=False,
    )

    return parser.parse_args()


if __name__ == "__main__":

    parser = ArgumentParser()
    args = parse_arguments(parser)

    model_path = args.model_path
    output_file_dir = args.output_file_dir
    job_adverts_filename = args.job_adverts_filename

    job_ner = JobNER()
    nlp = job_ner.load_model(
        model_path, s3_download=False if args.use_local_model else True
    )
    labels = nlp.get_pipe("ner").labels

    s3 = get_s3_resource()

    model_train_info = load_json_dict(os.path.join(model_path, "train_details.json"))
    seen_job_ids_dict = model_train_info["seen_job_ids"]
    train_job_ids = set(
        [
            v["job_id"]
            for k, v in seen_job_ids_dict.items()
            if v["train/test"] == "train"
        ]
    )
    job_adverts = load_s3_data(s3, bucket_name, job_adverts_filename)

    predicted_skills = {}
    for job_id, job_info in tqdm(job_adverts.items()):
        job_advert_text = job_info["description"].replace("\n", " ")
        pred_ents = job_ner.predict(job_advert_text)
        skills = {label: [] for label in labels}
        for ent in pred_ents:
            skills[ent["label"]].append(job_advert_text[ent["start"] : ent["end"]])
        skills["Train_flag"] = True if job_id in train_job_ids else False
        predicted_skills[job_id] = skills

    output_dict = {
        "model_path": model_path,
        "job_adverts_filename": job_adverts_filename,
        "predictions": predicted_skills,
    }
    date_stamp = str(date.today().date()).replace("-", "")
    save_to_s3(
        s3,
        bucket_name,
        output_dict,
        os.path.join(output_file_dir, f"{date_stamp}_skills.json"),
    )
