"""Script to map extracted skills from NER model to
taxonomy skills."""
##############################################################
from ojd_daps_skills import config, bucket_name
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
)
from ojd_daps_skills.pipeline.skill_ner_mapping.skill_ner_mapper_utils import (
    preprocess_skill,
)

from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer
import re
import time
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

##############################################################


class BertVectorizer:
    """
    Use a pretrained transformers model to embed skills.
    In this form so it can be used as a step in the pipeline.
    """

    def __init__(
        self,
        bert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        multi_process=True,
        batch_size=32,
    ):
        self.bert_model_name = bert_model_name
        self.multi_process = multi_process
        self.batch_size = batch_size

    def fit(self, *_):
        self.bert_model = SentenceTransformer(self.bert_model_name)
        self.bert_model.max_seq_length = 512
        return self

    def transform(self, texts):
        print(f"Getting embeddings for {len(texts)} texts ...")
        t0 = time.time()
        if self.multi_process:
            print(".. with multiprocessing")
            pool = self.bert_model.start_multi_process_pool()
            self.embedded_x = self.bert_model.encode_multi_process(
                texts, pool, batch_size=self.batch_size
            )
            self.bert_model.stop_multi_process_pool(pool)
        else:
            self.embedded_x = self.bert_model.encode(texts, show_progress_bar=True)
        print(f"Took {time.time() - t0} seconds")
        return self.embedded_x


class SkillMapper:
    """
    Class to map extracted skills from NER model to a skills taxonomy.
    Attributes
    ----------
    taxonomy (str): name of taxonomy to be mapped to
    taxonomy_file_name (str): file name of taxonomy to be mapped to
    skill_name_col (str): skill column name
    skill_desc_col (str): skill description name
    bert_model_name (str): name of sentence transformer
    multi_process (bool): if vectoriser will multi_process
    batch_size (int): batch size
    ojo_skills_file_name (str): file name of extract ojo skills from ner model
    ----------
    Methods
    ----------
    get_taxonomy_skills(taxonomy_file_name):
        loads taxonomy from file and converts
        to dict where key is taxonomy skill id and values are skill and skill description.
    get_ojo_skills(ojo_skills_file_name):
        loads extracted skills from NER model.
    preprocess_ojo_skills(ojo_skills):
        preprocess skills extracted OJO skills from NER model.
    preprocess_taxonomy_skills(taxonomy_skills):
        preprocesses taxonomy skills.
    load_bert:
        loads bert vectoriser.
    fit_transform(skills):
        fits and transforms skills.
    map_skills(taxonomy, taxonomy_file_name, ojo_skills_file_name):
        loads taxonomy and OJO skills; preprocesses skills; embeds
        and maps OJO onto taxonomy skills based on cosine similarity.
    """

    def __init__(
        self,
        taxonomy: "esco",
        taxonomy_file_name: "escoe_extension/inputs/data/esco/skills_en.csv",
        skill_name_col: "preferredLabel",
        skill_desc_col: "description",
        bert_model_name: "sentence-transformers/paraphrase-MiniLM-L6-v2",
        multi_process: True,
        batch_size: 32,
        ojo_skills_file_name: config["ojo_skills_ner_path"],
    ):
        self.taxonomy = taxonomy
        self.taxonomy_file_name = taxonomy_file_name
        self.skill_name_col = skill_name_col
        self.skill_desc_col = skill_desc_col
        self.bert_model_name = bert_model_name
        self.multi_process = multi_process
        self.batch_size = batch_size
        self.ojo_skills_file_name = ojo_skills_file_name

    def get_taxonomy_skills(self, taxonomy_file_name):
        self.taxonomy_skills = load_s3_data(
            get_s3_resource(), bucket_name, self.taxonomy_file_name
        )
        self.taxonomy_skills["skill_id"] = [
            self.taxonomy + "_" + str(i) for i in self.taxonomy_skills.index
        ]
        self.taxonomy_skills_dict = self.taxonomy_skills.set_index("skill_id")[
            [self.skill_name_col, self.skill_desc_col]
        ].T.to_dict()

        return self.taxonomy_skills_dict

    def get_ojo_skills(self, ojo_skills_file_name):
        self.ojo_skills = load_s3_data(
            get_s3_resource(), bucket_name, self.ojo_skills_file_name
        )
        return self.ojo_skills

    def preprocess_ojo_skills(self, ojo_skills):
        self.ojo_job_ids = list(self.ojo_skills["predictions"].keys())
        self.clean_ojo_skills = dict()

        for ojo_job_id in self.ojo_job_ids:
            ojo_job_skills = self.ojo_skills["predictions"][ojo_job_id]["SKILL"]
            if ojo_job_skills != []:
                self.clean_ojo_skills[ojo_job_id] = list(
                    set([preprocess_skill(skill) for skill in ojo_job_skills])
                )

        return self.clean_ojo_skills

    def preprocess_taxonomy_skills(self, taxonomy_skills_dict):
        self.taxonomy_ids = list(self.taxonomy_skills_dict.keys())
        for skill in self.taxonomy_ids:
            self.taxonomy_skills_dict[skill]["clean_skills"] = preprocess_skill(
                self.taxonomy_skills_dict[skill][self.skill_name_col]
            )
        return self.taxonomy_skills_dict

    def load_bert(self):
        self.bert_vectorizer = BertVectorizer(
            bert_model_name=self.bert_model_name,
            multi_process=self.multi_process,
            batch_size=self.batch_size,
        )
        return self.bert_vectorizer.fit()

    def transform(self, skills):
        # Load BERT model and transform skill
        self.skills_vec = self.bert_vectorizer.transform(skills)
        return self.skills_vec

    def map_skills(self, taxonomy, taxonomy_file_name, ojo_skills_file_name):
        self.taxonomy_skills = self.get_taxonomy_skills(self.taxonomy_file_name)
        self.ojo_skills = self.get_ojo_skills(self.ojo_skills_file_name)
        self.bert_vectorizer = self.load_bert()

        if self.taxonomy_skills:
            self.clean_taxonomy_skills = self.preprocess_taxonomy_skills(
                self.taxonomy_skills
            )
            self.taxonomy_skills_embeddings = self.bert_vectorizer.transform(
                [
                    self.clean_taxonomy_skills[skill]["clean_skills"]
                    for skill in self.taxonomy_ids
                ]
            )

            clean_ojo_skills = self.preprocess_ojo_skills(self.ojo_skills)
            clean_ojo_skill_ids = list(clean_ojo_skills.keys())

            self.skill_mapper_dict = dict()
            for skill in clean_ojo_skill_ids:
                top_tax_skills = []
                top_tax_scores = []
                clean_ojo_skill_embeddings = self.bert_vectorizer.transform(
                    clean_ojo_skills[skill]
                )
                ojo_taxonomoy_sims = cosine_similarity(
                    clean_ojo_skill_embeddings, self.taxonomy_skills_embeddings
                )

                for sim in ojo_taxonomoy_sims:
                    top_skill_ids = [
                        self.taxonomy_ids[i] for i in np.argsort(sim)[::-1][:5]
                    ]
                    top_tax_skills.append(
                        [
                            self.taxonomy_skills[i][self.skill_name_col]
                            for i in top_skill_ids
                        ]
                    )
                    top_tax_scores.append(
                        [float(i) for i in np.sort(sim)[::-1][:5]]
                    )  # so its JSON serializable
                self.skill_mapper_dict[skill] = {
                    "ojo_ner_skills": clean_ojo_skills[skill],
                    self.taxonomy + "_taxonomy_skills": top_tax_skills,
                    self.taxonomy + "_taxonomy_scores": top_tax_scores,
                }
            return self.skill_mapper_dict
        else:
            print("Warning! No taxonomy to map skill spans to.")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--taxonomy", help="Name of taxonomy to be mapped to.", default="esco",
    )

    parser.add_argument(
        "--taxonomy_fn",
        help="Name of taxonomy skills file name to be mapped to.",
        default="escoe_extension/inputs/data/esco/skills_en.csv",
    )

    parser.add_argument(
        "--ojo_skill_fn",
        help="Name of ojo skills file name to be mapped to.",
        default=config["ojo_skills_ner_path"],
    )

    args = parser.parse_args()

    taxonomy = args.taxonomy
    taxonomy_file_name = args.taxonomy_fn
    ojo_skill_file_name = args.ojo_skill_fn

    skill_mapper = SkillMapper(
        taxonomy=taxonomy,
        taxonomy_file_name=taxonomy_file_name,
        skill_name_col="preferredLabel",
        skill_desc_col="description",
        bert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        multi_process=True,
        batch_size=32,
        ojo_skills_file_name=config["ojo_skills_ner_path"],
    )

    skills_to_taxonomy = skill_mapper.map_skills(
        taxonomy, taxonomy_file_name, ojo_skill_file_name
    )

    skill_mapper_file_name = (
        ojo_skill_file_name.split("/")[-1].split(".")[0] + "_to_" + taxonomy + ".json"
    )

    save_to_s3(
        get_s3_resource(),
        bucket_name,
        skills_to_taxonomy,
        os.path.join(config["ojo_skills_ner_mapping_dir"], skill_mapper_file_name),
    )
