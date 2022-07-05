"""
This script contains the class needed to train, predict, load, and save and NER model.

A model can be trained by running this script:

python ojd_daps_skills/pipeline/skill_ner/ner_spacy.py
    --labelled_data_s3_folder "escoe_extension/outputs/skill_span_labels/"
    --label_metadata_filename "escoe_extension/outputs/data/skill_ner/label_chunks/20220624_0_sample_labelling_metadata.json"
    --convert_multiskill
    --train_prop 0.8
    --drop_out 0.3
    --num_its 50

This will save out the model in a time stamped folder,
e.g. `outputs/models/ner_model/20220629/`, it also saves out the evaluation results
and some general information about the model training in the file
`outputs/models/ner_model/20220629/train_details.json`.

By default this won't sync the newly trained model to S3, but by adding
`--save_s3` it will sync the `outputs/models/ner_model/20220629/` to S3.

Additionally you can use the class in this script to load a model and make predictions:

from ojd_daps_skills.pipeline.skill_ner.ner_spacy import JobNER
job_ner = JobNER()
nlp = job_ner.load_model('outputs/models/ner_model/20220630/', s3_download=True)
text = "The job involves communication and maths skills"
pred_ents = job_ner.predict(text)

"""

import random
import json
import pandas as pd
import os
from datetime import datetime as date
from argparse import ArgumentParser

from spacy.util import minibatch, compounding
from spacy.training.example import Example
import spacy
from spacy import displacy
from tqdm import tqdm
from nervaluate import Evaluator

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    load_s3_json,
    get_s3_data_paths,
    save_json_dict,
)
from ojd_daps_skills.pipeline.skill_ner.ner_spacy_utils import (
    clean_entities_text,
)
from ojd_daps_skills import bucket_name


class JobNER(object):
    """

    Attributes
    ----------
    BUCKET_NAME : str
        The bucket name where you will store data and the model.
    S3_FOLDER : str
        The S3 folder where the labelled data is.
    label_metadata_filename : str
        The S3 location where the metadata for the labelled data sample is.
    convert_multiskill : bool
        Where you want to convert all MULTISKILL spans to SKILL (True) or not (False)
    train_prop : float
        What proportion of the data do you want to use in the train split.

    Methods
    -------
    load_data():
        Load the data, remove duplicates, and process it into a form
        needed for training.
    get_test_train(data):
        Split the data into a test and training set.
    prepare_model():
        Prepare to train the NER model.
    train(train_data, print_losses=True, drop_out=0.3, num_its=30):
        Train the NER model using the training data.
    predict(job_text):
        Given a job advert text use the model to predict skills using
        the NER model
    display_prediction(job_text):
        Use displacy to render a nicely formatted job advert with
        predicted skill spans highlighted
    evaluate(data):
        Evaluate the model using the hold out test data.
    score(results_summary):
        Return a single evaluation score (F1).
    save_model(output_folder, save_s3=False):
        Save the model locally with the option of also saving it to S3.
    load_model(model_folder, s3_download=True):
        Load a model with the option of first downloading it locally from S3.
    """

    def __init__(
        self,
        BUCKET_NAME="open-jobs-lake",
        S3_FOLDER="escoe_extension/outputs/skill_span_labels/",
        label_metadata_filename="escoe_extension/outputs/data/skill_ner/label_chunks/20220624_0_sample_labelling_metadata.json",
        convert_multiskill=True,
        train_prop=0.8,
    ):
        self.BUCKET_NAME = BUCKET_NAME
        self.S3_FOLDER = S3_FOLDER
        self.label_metadata_filename = label_metadata_filename
        self.convert_multiskill = convert_multiskill
        self.train_prop = train_prop

    def process_data(self, job_advert_labels, all_labels):
        """
        Process the raw labelled data about job adverts, some text cleaning is needed,
        but we need to be careful to make sure span indices are still correct
        """

        text = job_advert_labels["task"]["data"]["text"]
        ent_tags = job_advert_labels["result"]

        ent_list = []
        for ent_tag in ent_tags:
            ent_tag_value = ent_tag["value"]
            label = ent_tag_value["labels"][0]
            if self.convert_multiskill:
                label = "SKILL" if label == "MULTISKILL" else label
            ent_list.append((ent_tag_value["start"], ent_tag_value["end"], label))
            if label not in all_labels:
                all_labels.add(label)

        # The entity list is in the order labelled not in
        # character order
        ent_list.sort(key=lambda y: y[0])

        text, ent_list = clean_entities_text(text, ent_list)

        return text, ent_list, all_labels

    def load_data(self):
        s3 = get_s3_resource()
        file_names = get_s3_data_paths(s3, self.BUCKET_NAME, self.S3_FOLDER, "*")
        file_names.remove(self.S3_FOLDER)

        # Find the label ID of job adverts we want to include
        label_meta = []
        for file_name in file_names:
            job_advert_labels = load_s3_json(s3, self.BUCKET_NAME, file_name)
            label_meta.append(
                {
                    "created_username": job_advert_labels["created_username"],
                    "id": job_advert_labels["id"],
                    "task_ids": job_advert_labels["task"]["id"],
                    "updated_at": job_advert_labels["updated_at"],
                    "task_is_labeled": job_advert_labels["task"]["is_labeled"],
                    "was_cancelled": job_advert_labels["was_cancelled"],
                }
            )
        label_meta = pd.DataFrame(label_meta)

        self.sorted_df = label_meta.sort_values(by=["updated_at"], ascending=False)
        self.sorted_df = self.sorted_df[~self.sorted_df["was_cancelled"]]
        self.sorted_df.drop_duplicates(subset=["task_ids"], keep="first", inplace=True)
        self.keep_label_ids = self.sorted_df["id"].tolist()
        print(f"We will be using data from {len(self.keep_label_ids)} job adverts")

        # Link the task ID to the actual job adverts ID using the metadata dictionary
        label_job_id_dict = load_s3_data(
            s3, self.BUCKET_NAME, self.label_metadata_filename
        )
        label_job_id_dict = {int(k): v for k, v in label_job_id_dict.items()}
        self.sorted_df["job_id"] = self.sorted_df["task_ids"].map(label_job_id_dict)
        # Keep a record of the job adverts the model has seen
        self.seen_job_ids = (
            self.sorted_df[["job_id", "task_ids"]]
            .set_index("task_ids")
            .to_dict(orient="index")
        )

        data = []
        self.all_labels = set()
        for file_name in file_names:
            job_advert_labels = load_s3_json(s3, self.BUCKET_NAME, file_name)
            if job_advert_labels["id"] in self.keep_label_ids:
                text, ent_list, self.all_labels = self.process_data(
                    job_advert_labels, self.all_labels
                )
                data.append(
                    (
                        text,
                        {"entities": ent_list},
                        {
                            "task_id": job_advert_labels["task"]["id"],
                            "job_ad_id": self.seen_job_ids[
                                job_advert_labels["task"]["id"]
                            ],
                            "label_id": job_advert_labels["id"],
                        },
                    )
                )

        return data

    def get_test_train(self, data):

        train_n = round(len(data) * self.train_prop)

        random.seed(42)
        random.shuffle(data)

        train_data = data[0:train_n]
        test_data = data[train_n:]

        for _, _, d in train_data:
            self.seen_job_ids[d["task_id"]]["train/test"] = "train"
        for _, _, d in test_data:
            self.seen_job_ids[d["task_id"]]["train/test"] = "test"

        return train_data, test_data

    def prepare_model(self):
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("ner")
        self.nlp.begin_training()

        # self.nlp = spacy.load("en_core_web_sm")

        # Getting the ner component
        ner = self.nlp.get_pipe("ner")

        # Add the new labels to ner
        for label in self.all_labels:
            ner.add_label(label)

        # Resume training
        self.optimizer = self.nlp.resume_training()
        move_names = list(ner.move_names)

    def train(self, train_data, print_losses=True, drop_out=0.3, num_its=30):
        """
        See https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/
        for the inspiration for this function.
        """
        self.train_data_length = len(train_data)
        self.drop_out = drop_out
        self.num_its = num_its
        # List of pipes you want to train
        pipe_exceptions = ["ner"]
        # List of pipes which should remain unaffected in training
        other_pipes = [
            pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions
        ]

        # Begin training by disabling other pipeline components
        with self.nlp.disable_pipes(*other_pipes):
            sizes = compounding(1.0, 4.0, 1.001)
            # Training for num_its iterations
            for itn in tqdm(range(num_its)):
                # shuffle examples before training
                random.shuffle(train_data)
                # batch up the examples using spaCy's minibatch
                batches = minibatch(train_data, size=sizes)
                # Dictionary to store losses
                losses = {}
                for batch in batches:
                    # Calling update() over the iteration
                    for text, annotation, _ in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotation)
                        # Update the model
                        self.nlp.update(
                            [example], sgd=self.optimizer, losses=losses, drop=drop_out
                        )
                if print_losses:
                    print(losses)

        return self.nlp

    def predict(self, job_text):
        doc = self.nlp(job_text)
        pred_ents = []
        for ent in doc.ents:
            pred_ents.append(
                {"label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            )
        return pred_ents

    def display_prediction(self, job_text):
        doc = self.nlp(job_text)
        displacy.render(doc, style="ent")

    def evaluate(self, data):

        truth = []
        preds = []
        for text, true_ents, _ in data:
            ad_truth = []
            for b, e, l in true_ents["entities"]:
                ad_truth.append({"label": l, "start": b, "end": e})
            truth.append(ad_truth)
            preds.append(self.predict(text))

        evaluator = Evaluator(truth, preds, tags=self.all_labels)
        results_all, results_per_tag = evaluator.evaluate()

        results_summary = {}

        all_dict = {}
        for ev_type in ["f1", "precision", "recall"]:
            all_dict[ev_type] = results_all["partial"][ev_type]
        results_summary["All"] = all_dict

        for label, lab_res in results_per_tag.items():
            lab_dict = {}
            for ev_type in ["f1", "precision", "recall"]:
                lab_dict[ev_type] = lab_res["partial"][ev_type]
            results_summary[label] = lab_dict

        self.evaluation_results = {
            "eval_data_length": len(data),
            "results_summary": results_summary,
            "results_all": results_all,
            "results_per_tag": results_per_tag,
        }
        return self.evaluation_results

    def score(self, results_summary):
        return results_summary["All"]["f1"]

    def save_model(self, output_folder, save_s3=False):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.nlp.to_disk(output_folder)

        # Output the training details of the model inc evaluation results (if done)
        try:
            model_details_dict = self.evaluation_results
        except AttributeError:
            model_details_dict = {}
        model_details_dict.update(
            {
                "BUCKET_NAME": self.BUCKET_NAME,
                "S3_FOLDER": self.S3_FOLDER,
                "convert_multiskill": self.convert_multiskill,
                "train_prop": self.train_prop,
                "labels": list(self.all_labels),
                "train_data_length": self.train_data_length,
                "drop_out": self.drop_out,
                "num_its": self.num_its,
                "seen_job_ids": self.seen_job_ids,
            }
        )
        save_json_dict(
            model_details_dict, os.path.join(output_folder, "train_details.json")
        )
        if save_s3:
            # Sync this to S3
            cmd = f"aws s3 sync {output_folder} s3://{self.BUCKET_NAME}/escoe_extension/{output_folder}"
            os.system(cmd)

    def load_model(self, model_folder, s3_download=True):
        if s3_download:
            # Download this model from S3
            cmd = f"aws s3 sync s3://{self.BUCKET_NAME}/escoe_extension/{model_folder} {model_folder}"
            os.system(cmd)
        else:
            print("Loading the model from a local location")

        try:
            self.nlp = spacy.load(model_folder)
        except OSError:
            print(
                "Model not found locally - you may need to download it from S3 (set s3_download to True)"
            )
        return self.nlp


def parse_arguments(parser):

    parser.add_argument(
        "--labelled_data_s3_folder",
        help="The S3 location of the labelled job adverts",
        default="escoe_extension/outputs/skill_span_labels/",
    )
    parser.add_argument(
        "--label_metadata_filename",
        help="The S3 path to labelling metadata for the job adverts that were labelled",
        default="escoe_extension/outputs/data/skill_ner/label_chunks/20220624_0_sample_labelling_metadata.json",
    )
    parser.add_argument(
        "--convert_multiskill",
        help="Convert the MULTISKILL labels to SKILL labels",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--train_prop",
        help="The proportion of labelled data to use in the training set",
        default=0.8,
    )
    parser.add_argument(
        "--drop_out",
        help="The drop out rate for the model",
        default=0.3,
    )
    parser.add_argument(
        "--num_its",
        help="The number of iterations in the training process",
        default=50,
    )
    parser.add_argument(
        "--save_s3",
        help="Save the model to S3",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":

    # Train a model

    parser = ArgumentParser()
    args = parse_arguments(parser)

    job_ner = JobNER(
        BUCKET_NAME=bucket_name,
        S3_FOLDER=args.labelled_data_s3_folder,
        label_metadata_filename=args.label_metadata_filename,
        convert_multiskill=args.convert_multiskill,
        train_prop=float(args.train_prop),
    )
    data = job_ner.load_data()
    train_data, test_data = job_ner.get_test_train(data)
    job_ner.prepare_model()
    nlp = job_ner.train(
        train_data,
        print_losses=True,
        drop_out=float(args.drop_out),
        num_its=int(args.num_its),
    )

    from datetime import datetime as date

    date_stamp = str(date.today().date()).replace("-", "")
    output_folder = f"outputs/models/ner_model/{date_stamp}"
    results = job_ner.evaluate(test_data)
    job_ner.save_model(output_folder, args.save_s3)
