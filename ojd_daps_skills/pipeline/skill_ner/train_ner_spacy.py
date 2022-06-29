import random
import json
import pandas as pd
import os
from datetime import datetime as date

from spacy.util import minibatch, compounding
from spacy.training.example import Example
import spacy
from spacy import displacy
from tqdm import tqdm
from nervaluate import Evaluator

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    save_to_s3,
    load_s3_json,
    get_s3_data_paths,
    save_json_dict,
)
from ojd_daps_skills.pipeline.skill_ner.train_ner_spacy_utils import (
    edit_ents,
    fix_formatting_entities,
)


class JobNER(object):
    """ """

    def __init__(
        self,
        BUCKET_NAME="open-jobs-lake",
        S3_FOLDER="escoe_extension/outputs/skill_span_labels/",
        convert_multiskill=True,
        train_prop=0.8,
    ):
        self.BUCKET_NAME = BUCKET_NAME
        self.S3_FOLDER = S3_FOLDER
        self.convert_multiskill = convert_multiskill
        self.train_prop = train_prop

    def process_data(self, job_advert_labels, all_labels):
        """
        Use the raw labelled data about job adverts t
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

        text, ent_list = fix_formatting_entities(text, ent_list)

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
                            "job_ad_id": job_advert_labels["task"]["id"],
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

        return train_data, test_data

    def prepare_blank_model(self):
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("ner")
        self.nlp.begin_training()

        # Getting the ner component
        ner = self.nlp.get_pipe("ner")

        # Add the new labels to ner
        for label in self.all_labels:
            ner.add_label(label)

        # Resume training
        self.optimizer = self.nlp.resume_training()
        move_names = list(ner.move_names)

    def train(self, train_data, print_losses=True, drop_out=0.3, num_its=30):

        # For an output file
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
                    #              texts, annotations = zip(*batch)
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

    def save_model(self, output_folder, output_details=True):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.nlp.to_disk(output_folder)

        if output_details:
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
                }
            )
            save_json_dict(
                model_details_dict, os.path.join(output_folder, "train_details.json")
            )

    def load_model(self, model_folder):
        self.nlp = spacy.load(model_folder)
        return self.nlp


if __name__ == "__main__":

    job_ner = JobNER(
        BUCKET_NAME="open-jobs-lake",
        S3_FOLDER="escoe_extension/outputs/skill_span_labels/",
        convert_multiskill=True,
        train_prop=0.8,
    )
    data = job_ner.load_data()
    train_data, test_data = job_ner.get_test_train(data)
    job_ner.prepare_blank_model()
    nlp = job_ner.train(train_data, print_losses=True, drop_out=0.3, num_its=50)

    from datetime import datetime as date

    date_stamp = str(date.today().date()).replace("-", "")
    output_folder = f"outputs/models/ner_model/{date_stamp}"
    results = job_ner.evaluate(test_data)
    job_ner.save_model(output_folder)
