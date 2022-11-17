"""
This script contains the class needed to train, predict, load, and save and NER model.

A model can be trained by running this script:

python ojd_daps_skills/pipeline/skill_ner/ner_spacy.py
    --labelled_date_filename "escoe_extension/outputs/labelled_job_adverts/combined_labels_20220824.json"
    --convert_multiskill
    --train_prop 0.8
    --drop_out 0.3
    --num_its 50

This will save out the model in a time stamped folder,
e.g. `outputs/models/ner_model/2022XXXX/`, it also saves out the evaluation results
and some general information about the model training in the file
`outputs/models/ner_model/2022XXXX/train_details.json`.

By default this won't sync the newly trained model to S3, but by adding
`--save_s3` it will sync the `outputs/models/ner_model/2022XXXX/` to S3.

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
import pickle

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
from ojd_daps_skills.pipeline.skill_ner.multiskill_utils import MultiskillClassifier
from ojd_daps_skills import bucket_name, logger, PROJECT_DIR


class JobNER(object):
    """

    Attributes
    ----------
    BUCKET_NAME : str
        The bucket name where you will store data and the model.
    labelled_date_filename : str
        The S3 file where the labelled data is.
    convert_multiskill : bool
        Where you want to convert all MULTISKILL spans to SKILL (True) or not (False)
        for the training of the NER model
    train_prop : float
        What proportion of the data do you want to use in the train split.

    Methods
    -------
    load_data():
        Load the data, remove duplicates, and process it into a form
        needed for training.
    get_test_train(data):
        Split the data into a test and training set.
    train_multiskill_classifier(train_data, test_data):
        Uses the clean labelled test and train set to train a simple
        classifier to predict whether a skill is multi or single skill.
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
        labelled_date_filename="escoe_extension/outputs/labelled_job_adverts/combined_labels_20220824.json",
        convert_multiskill=True,
        train_prop=0.8,
    ):
        self.BUCKET_NAME = BUCKET_NAME
        self.labelled_date_filename = labelled_date_filename
        self.convert_multiskill = convert_multiskill
        self.train_prop = train_prop

    def process_data(self, job_advert_labels, all_labels):
        """
        Process the raw labelled data about job adverts, some text cleaning is needed,
        but we need to be careful to make sure span indices are still correct

        Parameters
        ----------
        job_advert_labels : dict
            The raw label-studio labelled data for one job advert
        all_labels : list
            The list of all labels given to entities

        Returns
        ------
        text : str
            The cleaned job advert text
        ent_list : list
            The entity span list (modified after cleaning the text)
            this is in the form [(start_char, end_char, label),...]
        all_labels : list
            The list of all labels given to entities
        """

        text = job_advert_labels["text"]
        ent_tags = job_advert_labels["labels"]

        ent_list = []
        for ent_tag in ent_tags:
            ent_tag_value = ent_tag["value"]
            label = ent_tag_value["labels"][0]
            ent_list.append((ent_tag_value["start"], ent_tag_value["end"], label))
            if label not in all_labels:
                all_labels.add(label)

        # The entity list is in the order labelled not in
        # character order
        ent_list.sort(key=lambda y: y[0])

        text, ent_list = clean_entities_text(text, ent_list)

        return text, ent_list, all_labels

    def load_data(self):
        """
        Load all the labelled job adverts from the label-studio output in S3.
        If more than one person has labelled a job advert only the latest labels
        will be used.

        Returns
        ------
        data : list

            The job adverts and the entities within them in a format suitable for Spacy
            training, i.e. a list of tuples
            [(text, {"entities": [(0,4,"SKILL"),...]}, {"job_ad_id": 'as34d'}),...]
        """

        s3 = get_s3_resource()
        job_advert_labels = load_s3_json(
            s3, self.BUCKET_NAME, self.labelled_date_filename
        )
        logger.info(f"We will be using data from {len(job_advert_labels)} job adverts")
        self.seen_job_ids = {k: {} for k in job_advert_labels.keys()}

        data = []
        self.all_labels = set()
        for job_ad_id, label_data in job_advert_labels.items():
            text, ent_list, self.all_labels = self.process_data(
                label_data, self.all_labels
            )
            data.append(
                (
                    text,
                    {"entities": ent_list},
                    {
                        "job_ad_id": job_ad_id,
                    },
                )
            )

        return data

    def get_test_train(self, data):
        """
        Split the data into a training and test set, and keep a record
        of which job ids were used in each.
        """

        train_n = round(len(data) * self.train_prop)

        random.seed(42)
        random.shuffle(data)

        train_data = data[0:train_n]
        test_data = data[train_n:]

        for _, _, d in train_data:
            self.seen_job_ids[d["job_ad_id"]]["train/test"] = "train"
        for _, _, d in test_data:
            self.seen_job_ids[d["job_ad_id"]]["train/test"] = "test"

        return train_data, test_data

    def multiskill_conversion(self, data):
        """
        Convert rest of the multiskill labels to skills if desired
        """
        data_cleaned = []
        for text, ents, meta in data:
            ents_cleaned = []
            for start, end, label in ents["entities"]:
                label = "SKILL" if label == "MULTISKILL" else label
                ents_cleaned.append((start, end, label))
            data_cleaned.append((text, {"entities": ents_cleaned}, meta))
        return data_cleaned

    def prepare_model(self):
        """
        Prepare a Spacy model to have it's NER component trained
        """
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("ner")
        self.nlp.begin_training()

        # self.nlp = spacy.load("en_core_web_sm")

        # Getting the ner component
        ner = self.nlp.get_pipe("ner")

        # Add the new labels to ner (don't train the MULTISKILL)
        self.train_labels = self.all_labels.copy()
        if self.convert_multiskill:
            self.train_labels.remove("MULTISKILL")

        for label in self.train_labels:
            ner.add_label(label)

        # Resume training
        self.optimizer = self.nlp.resume_training()
        move_names = list(ner.move_names)

    def train_multiskill_classifier(self, train_data, test_data):
        """
        Uses the clean labelled test and train set (same as the NER model will use)
        to train a simple classifier to predict whether a skill is multi or single skill.
        Also gets train and test scores for the output.
        """

        def separate_labels(data, ms_classifier):
            """
            Only needed in this function
            """
            skill_ent_list = []
            multiskill_ent_list = []
            for text, ents, _ in data:
                for start, end, label in ents["entities"]:
                    if label == "SKILL":
                        skill_ent_list.append(text[start:end])
                    if label == "MULTISKILL":
                        multiskill_ent_list.append(text[start:end])
            X, y = ms_classifier.create_training_data(
                skill_ent_list, multiskill_ent_list
            )
            return X, y

        self.ms_classifier = MultiskillClassifier()
        X_train, y_train = separate_labels(train_data, self.ms_classifier)
        X_test, y_test = separate_labels(test_data, self.ms_classifier)

        self.ms_classifier.fit(X_train, y_train)

        self.ms_classifier_train_evaluation = self.ms_classifier.score(X_train, y_train)
        self.ms_classifier_test_evaluation = self.ms_classifier.score(X_test, y_test)

    def train(
        self,
        train_data,
        test_data,
        print_losses=True,
        drop_out=0.1,
        num_its=30,
        learn_rate=0.001,
    ):
        """
        Train a Spacy model for the NER task
        See https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/
        for the inspiration for this function.

        Parameters
        ----------
        train_data : list
            A list of tuples for each job advert in the training set, e.g.
            [(text, {"entities": [(0,4,"SKILL"),...]}, {"job_ad_id": ...}),...]
            only the first two elements of the tuples are needed
        print_losses : bool
            Print the losses as you train (can be useful in experimentation to check you have converged)
        drop_out : float
            Drop out rate for the training
        num_its : int
            Number of iterations to train the model
        learn_rate : float
            Learning rate for the training

        Returns
        ------
        nlp : Spacy language model
            A nlp language model with a NER component to recognise skill entities
        """

        # Before converting multiskills to skill entities, train the multiskill classifier
        self.train_multiskill_classifier(train_data, test_data)

        if self.convert_multiskill:
            train_data = self.multiskill_conversion(train_data)

        self.train_data_length = len(train_data)
        self.train_num_ents = sum([len(t[1]["entities"]) for t in train_data])
        self.test_num_ents = sum([len(t[1]["entities"]) for t in test_data])
        self.drop_out = drop_out
        self.num_its = num_its
        self.learn_rate = learn_rate
        # List of pipes you want to train
        pipe_exceptions = ["ner"]
        # List of pipes which should remain unaffected in training
        other_pipes = [
            pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions
        ]

        self.optimizer.learn_rate = self.learn_rate

        # Begin training by disabling other pipeline components
        self.all_losses = []
        with self.nlp.disable_pipes(*other_pipes):
            sizes = compounding(1.0, 4.0, 1.001)
            # Training for num_its iterations
            for itn in tqdm(range(num_its)):
                # shuffle examples before training
                random.seed(itn)
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
                self.all_losses.append(losses["ner"])
                if print_losses:
                    logger.info(losses)

        return self.nlp

    def predict(self, job_text):
        """
        Predict the entities in a single job advert text
        Parameters
        ----------
        job_text : str

        Returns
        ------
        pred_ents : list of dicts
            The entity span predictions in the form
            e.g. [{"label": "SKILL", "start": start_entity_char, "end": end_entity_char}, ...]
        """

        doc = self.nlp(job_text)
        pred_ents = []
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                # Apply the classifier to see whether it's likely to be a multiskill
                if self.ms_classifier.predict(ent.text)[0] == 1:
                    ent.label_ = "MULTISKILL"
            if (len(ent.text) > 1) | (ent.text == "R") | (ent.text == "C"):
                pred_ents.append(
                    {"label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                )
        return pred_ents

    def display_prediction(self, job_text):
        doc = self.nlp(job_text)
        displacy.render(doc, style="ent")

    def evaluate(self, data):
        """
        For a dataset of text and entity truths, evaluate how well the model
        finds entities. Various metrics are outputted.
        """

        # if self.convert_multiskill:
        #     data = self.multiskill_conversion(data)

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

        output_folder = os.path.join(str(PROJECT_DIR), output_folder)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.nlp.to_disk(output_folder)
        pickle.dump(
            self.ms_classifier,
            open(os.path.join(output_folder, "ms_classifier.pkl"), "wb"),
        )

        # Output the training details of the model inc evaluation results (if done)
        try:
            model_details_dict = self.evaluation_results
        except AttributeError:
            model_details_dict = {}
        model_details_dict.update(
            {
                "BUCKET_NAME": self.BUCKET_NAME,
                "labelled_date_filename": self.labelled_date_filename,
                "convert_multiskill": self.convert_multiskill,
                "train_prop": self.train_prop,
                "labels": list(self.all_labels),
                "train_data_length": self.train_data_length,
                "train_num_ents": self.train_num_ents,
                "test_num_ents": self.test_num_ents,
                "drop_out": self.drop_out,
                "num_its": self.num_its,
                "learn_rate": self.learn_rate,
                "ms_classifier_train_evaluation": self.ms_classifier_train_evaluation,
                "ms_classifier_test_evaluation": self.ms_classifier_test_evaluation,
                "seen_job_ids": self.seen_job_ids,
                "losses": self.all_losses,
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
            if "escoe_extension/" in model_folder:
                s3_folder = model_folder
                model_folder = model_folder.split("escoe_extension/")[1]
            else:
                s3_folder = os.path.join("escoe_extension/", model_folder)
            # If we havent already downloaded it, do so
            if not os.path.exists(model_folder):
                # Download this model from S3
                cmd = f"aws s3 sync s3://{self.BUCKET_NAME}/{s3_folder} {model_folder}"
                os.system(cmd)
        else:
            logger.info("Loading the model from a local location")

        try:
            logger.info(f"Loading the model from {model_folder}")
            self.nlp = spacy.load(model_folder)
            self.ms_classifier = pickle.load(
                open(os.path.join(model_folder, "ms_classifier.pkl"), "rb")
            )
        except OSError:
            logger.info(
                "Model not found locally - you may need to download it from S3 (set s3_download to True)"
            )
        return self.nlp


def parse_arguments(parser):

    parser.add_argument(
        "--labelled_date_filename",
        help="The S3 location of the labelled job adverts",
        default="escoe_extension/outputs/labelled_job_adverts/combined_labels_20220824.json",
    )

    parser.add_argument(
        "--convert_multiskill",
        help="Convert the MULTISKILL labels to SKILL labels",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--train_prop",
        help="The proportion of labelled data to use in the training set",
        default=0.8,
    )
    parser.add_argument(
        "--drop_out",
        help="The drop out rate for the model",
        default=0.1,
    )
    parser.add_argument(
        "--num_its",
        help="The number of iterations in the training process",
        default=100,
    )
    parser.add_argument(
        "--learn_rate",
        help="The learning rate for the model",
        default=0.001,
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

    logger.info(f"multiskill arg: {args.convert_multiskill}")

    job_ner = JobNER(
        BUCKET_NAME=bucket_name,
        labelled_date_filename=args.labelled_date_filename,
        convert_multiskill=args.convert_multiskill,
        train_prop=float(args.train_prop),
    )
    data = job_ner.load_data()
    train_data, test_data = job_ner.get_test_train(data)

    job_ner.prepare_model()
    nlp = job_ner.train(
        train_data,
        test_data,
        print_losses=True,
        drop_out=float(args.drop_out),
        num_its=int(args.num_its),
        learn_rate=float(args.learn_rate),
    )

    from datetime import datetime as date

    date_stamp = str(date.today().date()).replace("-", "")
    output_folder = f"outputs/models/ner_model/{date_stamp}"
    results = job_ner.evaluate(test_data)
    job_ner.save_model(output_folder, args.save_s3)
