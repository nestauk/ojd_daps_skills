"""
This script contains the class needed to train a simple classifier to predict whether a skill entity is a skill (1) or not a skill (0)
"""
import sys
sys.path.append("/Users/india.kerlenesta/Projects/ojd_daps_extension/ojd_daps_skills/")

import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

from ojd_daps_skills.utils.bert_vectorizer import BertVectorizer
from ojd_daps_skills import logger, bucket_name
from ojd_daps_skills.getters.data_getters import load_file, save_json_dict
import pickle
import os
import logging


class SpanClassifier:
    def __init__(self, bert_model=BertVectorizer().fit(), verbose=True):
        self.bert_model = bert_model
        self.verbose = verbose
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)

    """
    Train a classifier to predict whether a skill entity is a skill or not
    """

    def transform(self, entity_list):
        """
        Given a list of entity texts, or a single one, turn this into a vector
        of basic features
        """
        if isinstance(entity_list, str):
            entity_list = [entity_list]

        entity_list_embedding = self.bert_model.transform(entity_list)

        return entity_list_embedding

    def create_training_data(self, skill_ent_list, nonskill_ent_list):
        """
        Sample the input data and transform it into a feature vector
        """
        num_each_class = min(len(skill_ent_list), len(nonskill_ent_list))

        # Use equal numbers of each class (skills is usually a lot bigger than nonskills)
        skill_ent_list_sampled = random.sample(skill_ent_list, num_each_class)
        nonskill_ent_list_sampled = random.sample(nonskill_ent_list, num_each_class)

        X = np.concatenate(
            [
                self.transform(skill_ent_list_sampled),
                self.transform(nonskill_ent_list_sampled),
            ]
        )

        y = ["skill"] * len(skill_ent_list_sampled) + ["nonskill"] * len(
            nonskill_ent_list_sampled
        )
        indices = [_ for _ in range(0, len(X))]

        return X, y, indices

    def split_training_data(self, X, y, indices, random_state, test_size=0.2):
        """Split the input data into a train-test split"""
        (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
        ) = train_test_split(X, y, indices, test_size=test_size, shuffle=True, random_state=random_state)
        logger.info(f"the train set size is {len(y_train)}")
        logger.info(f"the test set size is {len(y_test)}")

        return X_train, X_test, y_train, y_test, indices_train, indices_test

    def fit(self, X, y):
        """Fit the classifier"""
        self.lg = LogisticRegression().fit(X, y)
        return self.lg

    def predict(self, X):
        """Predict using the trained model"""

        if isinstance(X, str):
            X = self.transform(X)
        return self.lg.predict(X)

    def score(self, X, y):

        return self.lg.score(X, y)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return classification_report(y, y_pred, output_dict=True)

    def save_model(
        self, X_test, y_test, output_folder, bucket_name=bucket_name, save_s3=False
    ):

        results = self.evaluate(X_test, y_test)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pickle.dump(
            self.lg,
            open(os.path.join(output_folder, "skill_nonskill_classifier.pkl"), "wb"),
        )

        save_json_dict(results, os.path.join(output_folder, "train_details.json"))

        if save_s3:
            # Sync this to S3
            cmd = f"aws s3 sync {output_folder} s3://{bucket_name}/{output_folder}"
            os.system(cmd)


if __name__ == "__main__":

    skill_nonskill_labels = load_file(
        "escoe_extension/outputs/skill_nonskill_labels/skill_nonskill_labels.csv"
    )

    skill_ent_list = [
        str(i)
        for i in list(
            skill_nonskill_labels[skill_nonskill_labels["label"] == "skill"]["skill"]
        )
    ]
    nonskill_ent_list = [
        str(i)
        for i in list(
            skill_nonskill_labels[skill_nonskill_labels["label"] == "nonskill"]["skill"]
        )
    ]

    skill_classifier = SpanClassifier()
    X, y, indices = skill_classifier.create_training_data(
        skill_ent_list, nonskill_ent_list
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
    ) = skill_classifier.split_training_data(X, y, indices, test_size=0.2, random_state=45)

    skill_classifier.fit(X_train, y_train)

    from datetime import datetime as date

    date_stamp = str(date.today().date()).replace("-", "")
    output_folder = f"escoe_extension/outputs/models/span_classifier/{date_stamp}"

    # save to both public and private buckets
    skill_classifier.save_model(
        X_test, y_test, output_folder, bucket_name, save_s3=True
    )
    skill_classifier.save_model(
        X_test, y_test, output_folder, "open-jobs-indicators", save_s3=True
    )
