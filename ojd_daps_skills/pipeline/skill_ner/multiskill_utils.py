"""
This script contains the util functions needed to
- Train a simple classifier to predict whether a skill entity is a single (0) or multi skill (1) entity
- Rules to split multiskill entities into separate skills (by Julia Suter)
"""

import random
import re
from collections import defaultdict, Counter

from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    get_s3_data_paths,
    load_s3_json,
)
from ojd_daps_skills import bucket_name

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")


class MultiskillClassifier:
    """
    Train a classifier to predict whether a skill entity is a a multi-(1) or single-skill (0) entity
    """

    def transform(self, entity_list):
        """
        Given a list of entity texts, or a single one, turn this into a vector
        of basic features
        """

        if isinstance(entity_list, str):
            entity_list = [entity_list]

        entity_vec = []
        for entity in entity_list:
            entity_vec.append([len(entity), int(" and " in entity), int("," in entity)])

        return entity_vec

    def create_training_data(self, skill_ent_list, multiskill_ent_list):
        """
        Sample the input data and transform it into a feature vector
        """

        num_each_class = min(len(skill_ent_list), len(multiskill_ent_list))

        # Use equal numbers of each class (skill list is usually a lot bigger than multiskill)
        random.seed(42)
        skill_ent_list_sampled = random.sample(skill_ent_list, num_each_class)
        multiskill_ent_list_sampled = random.sample(multiskill_ent_list, num_each_class)

        X = self.transform(skill_ent_list_sampled) + self.transform(
            multiskill_ent_list_sampled
        )
        y = [0] * len(skill_ent_list_sampled) + [1] * len(multiskill_ent_list_sampled)

        return X, y

    def split_training_data(self, X, y, test_size=0.25):

        return train_test_split(X, y, test_size=test_size, random_state=0)

    def fit(self, X, y):
        self.clf = svm.SVC(kernel="linear", C=1, class_weight="balanced").fit(X, y)
        return self.clf

    def predict(self, X):
        if isinstance(X, str):
            X = self.transform(X)
        return self.clf.predict(X)

    def score(self, X, y):
        return self.clf.score(X, y)

    def evaluate(self, X, y):
        y_pred = self.predict(X)

        return classification_report(y, y_pred, target_names=["Skill", "Multiskill"])


def duplicate_object(parsed_sent):
    """
    Deal with 2 verbs + 1 object
    e.g.
    'using and providing clinical supervision' --> 'using clinical supervision' and 'providing clinical supervision'
    """

    for token in parsed_sent:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":

            has_AND = False
            has_second_verb = False
            has_dobj = False

            for child in token.children:

                if child.pos_ == "CCONJ" and child.lemma_ == "and":
                    has_AND = True

                if child.pos_ == "VERB" and child.dep_ == "conj":
                    has_second_verb = True
                    second_verb = child
                    first_verb = token

                    has_dobj = "dobj" in [o.dep_ for o in second_verb.subtree]

                    if has_dobj:
                        has_dobj = True
                        dobj = " ".join(
                            [
                                c.text
                                for c in second_verb.subtree
                                if c.text != second_verb.text
                            ]
                        )

            if has_AND and has_second_verb and has_dobj:

                first_skill = "{} {}".format(first_verb, dobj)
                second_skill = "{} {}".format(second_verb, dobj)

                return [first_skill, second_skill]

    return None


def split_on_and(text):
    """
    Split some text on the word 'and' and commas, but deal with oxford commas
    and consider 'and' in words (pad with space).
    e.g. don't split up "understanding"
    """
    # Get rid of any double spacing
    text = re.sub("\s\s+", " ", text)

    split_on = " and "

    # Sort out any combinations of 'and' and commas/semi-colons.
    text = text.replace(";", ",")
    text = (
        text.replace(", and ,", split_on)
        .replace(", and,", split_on)
        .replace(",and ,", split_on)
        .replace(", and ", split_on)
        .replace(" and ,", split_on)
    )
    text = (
        text.replace(",and,", split_on)
        .replace(" and,", split_on)
        .replace(",and ", split_on)
    )

    # Split on commas and 'and'
    text = text.replace(",", split_on).split(" and ")
    return [t.strip() for t in text]


def duplicate_verb(parsed_phrase):
    """
    Deal with 1 verb + 2 objects

    e.g. 'using smartphones and apps' --> 'using smartphones' and 'using apps'
    """

    for token in parsed_phrase:

        if token.pos_ == "VERB" and token.dep_ == "ROOT":

            has_AND = False
            has_dobj = False
            has_sec_obj = False

            for child in token.children:

                if child.dep_ == "dobj":
                    has_dobj = True

                    objects = " ".join(
                        [c.text for c in token.subtree if c.text != token.text]
                    )

                    split_objects = split_on_and(objects)

                    object_list = []
                    for split_skill in split_objects:
                        object_list.append(split_skill)

                    for subchild in child.children:

                        if subchild.pos_ == "CCONJ" and subchild.lemma_ == "and":
                            has_AND = True

                        if subchild.dep_ == "conj":
                            has_sec_obj = True

                    if has_AND and has_dobj and has_sec_obj:

                        skill_lists = [
                            "{} {}".format(token.text, split_skill)
                            for split_skill in object_list
                        ]

                        return skill_lists

    return None


def split_skill_mentions(parsed_phrase):
    """
    Deal with compounds, noun modifiers --> split noun phrases and complete

    e.g. 'written and oral communication skills' --> 'written skills' and 'oral communicaton skills'
    """

    for token in parsed_phrase:
        if (
            token.pos_ == "NOUN"
            and token.lemma_ == "skill"
            and token.idx == parsed_phrase[-1].idx
        ):

            has_AND = False

            root = [token for token in parsed_phrase if token.dep_ == "ROOT"]
            if root:
                root = root[0]

                for child in root.subtree:

                    if child.pos_ == "CCONJ" and child.lemma_ == "and":
                        has_AND = True

                if has_AND:
                    skill_def = " ".join(
                        [c.text for c in root.subtree if c.text != token.text]
                    )

                    split_skills = split_on_and(skill_def)

                    skill_lists = []
                    for split_skill in split_skills:
                        skill_lists.append("{} {}".format(split_skill, token.text))

                    return skill_lists
    return None


def split_multiskill(text, min_length=75):
    """
    For a single multiskill text, parse it and output it split into single skills.
    Will only be applied if the text is less than min_length characters.

    Ideally no entity would contain multiple sentences, but this does happen, so
    we should split by them.
    """
    rule_list = [duplicate_object, duplicate_verb, split_skill_mentions]

    # If there are fullstops then split by them
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences]

    if len(sentences) == 1:
        sentence = sentences[0]
        if len(sentence) <= min_length:
            parsed_text = nlp(sentence)
            for rule in rule_list:
                output = rule(parsed_text)
                if output is not None:
                    return output
    else:
        # Multiple sentences, try to split further, but
        # if not return the split sentence
        split_skills = []
        split_found = False
        for sentence in sentences:
            if len(sentence) <= min_length:
                parsed_text = nlp(sentence)

                for rule in rule_list:
                    output = rule(parsed_text)
                    if output is not None:
                        for skill in output:
                            split_skills.append(skill)
                        split_found = True
                        break
                if not split_found:
                    split_skills.append(sentence)
            else:
                split_skills.append(sentence)

        return split_skills
