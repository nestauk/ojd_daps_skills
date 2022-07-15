from ojd_daps_skills import config, bucket_name
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
    get_s3_data_paths,
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


class BertVectorizer:
    """
    Use a pretrained transformers model to embed skills.
    In this form so it can be used as a step in the pipeline.
    """

    def __init__(
        self,
        bert_model_name="jjzha/jobspanbert-base-cased",
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
    taxonomy_skill_list (list): list of skills at any taxonomy level
    skill_name_col (str): skill column name
    skill_desc_col (str): skill description name
    bert_model_name (str): name of sentence transformer
    multi_process (bool): if vectoriser will multi_process
    batch_size (int): batch size
    ojo_skills_file_name (str): file name of extract ojo skills from ner model
    ----------
    Methods
    ----------
    get_ojo_skills(ojo_skills_file_name):
        loads extracted skills from NER model.
    preprocess_ojo_skills(ojo_skills):
        preprocess skills extracted OJO skills from NER model.
    preprocess_taxonomy_skills(taxonomy_skills):
        preprocesses list of taxonomy skills.
    load_bert:
        loads bert vectoriser.
    transform(skills):
        transforms skills.
    map_skills(taxonomy, taxonomy_skill_list, ojo_skills_file_name):
        loads taxonomy and OJO skills; preprocesses skills; embeds
        and maps OJO onto taxonomy skills based on cosine similarity.
    """

    def __init__(
        self,
        taxonomy: "esco",
        skill_name_col: "preferredLabel",
        skill_desc_col: "description",
        bert_model_name: "jjzha/jobspanbert-base-cased",
        multi_process: True,
        batch_size: 32,
        ojo_skills_file_name: config["ojo_skills_ner_path"],
    ):
        self.taxonomy = taxonomy
        self.skill_name_col = skill_name_col
        self.skill_desc_col = skill_desc_col
        self.bert_model_name = bert_model_name
        self.multi_process = multi_process
        self.batch_size = batch_size
        self.ojo_skills_file_name = ojo_skills_file_name

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
                self.clean_ojo_skills[ojo_job_id] = {
                    "clean_skills": list(
                        set([preprocess_skill(skill) for skill in ojo_job_skills])
                    )
                }

        return self.clean_ojo_skills

    def preprocess_taxonomy_skills(self, taxonomy_skill_list):
        self.clean_taxonomy_skills = [
            preprocess_skill(skill) for skill in taxonomy_skill_list
        ]
        return self.clean_taxonomy_skills

    def load_bert(self):
        self.bert_vectorizer = BertVectorizer(
            bert_model_name=self.bert_model_name,
            multi_process=self.multi_process,
            batch_size=self.batch_size,
        )
        return self.bert_vectorizer.fit()

    def transform(self, skills):
        # Load BERT model and transform skill
        skills_vec = self.bert_vectorizer.transform(skills)
        return skills_vec

    def map_skills(self, taxonomy, taxonomy_skill_list, ojo_skills_file_name):

        self.ojo_skills = self.get_ojo_skills(self.ojo_skills_file_name)
        self.bert_vectorizer = self.load_bert()

        self.clean_taxonomy_skills = self.preprocess_taxonomy_skills(
            taxonomy_skill_list
        )
        self.taxonomy_skills_embeddings = self.bert_vectorizer.transform(
            self.clean_taxonomy_skills
        )

        clean_ojo_skills = self.preprocess_ojo_skills(self.ojo_skills)

        flat_clean_ojo_skills = list(
            itertools.chain(*[i["clean_skills"] for i in clean_ojo_skills.values()])
        )
        clean_ojo_skill_embeddings = self.bert_vectorizer.transform(
            flat_clean_ojo_skills
        )

        # map embeds onto clean_ojo_skills
        for id_, skill in clean_ojo_skills.items():
            ojo_skill_embeds = [
                clean_ojo_skill_embeddings[flat_clean_ojo_skills.index(s)]
                for s in skill["clean_skills"]
            ]
            skill["clean_ojo_embeds"] = ojo_skill_embeds

        skill_mapper_dict = dict()
        for id_, skill in clean_ojo_skills.items():
            top_tax_skills = []
            ojo_taxonomoy_sims = cosine_similarity(
                skill["clean_ojo_embeds"], self.taxonomy_skills_embeddings
            )
            top_skill_indxs = [
                list(np.argsort(sim)[::-1][:5]) for sim in ojo_taxonomoy_sims
            ]
            top_skill_scores = [np.sort(sim)[::-1][:5] for sim in ojo_taxonomoy_sims]

            skill_mapper_dict[id_] = {
                "ojo_ner_skills": skill["clean_skills"],
                "esco_taxonomy_skills": [
                    [self.clean_taxonomy_skills[i] for i in top_skills]
                    for top_skills in top_skill_indxs
                ],
                "esco_taxonomy_scores": [
                    [float(i) for i in top_skill_score]
                    for top_skill_score in top_skill_scores
                ],  # to navigate JSON serializable issues
            }
        return skill_mapper_dict


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--taxonomy", help="Name of taxonomy to be mapped to.", default="esco",
    )

    parser.add_argument(
        "--ojo_skill_fn",
        help="Name of ojo skills file name to be mapped to.",
        default=config["ojo_skills_ner_path"],
    )

    args = parser.parse_args()

    taxonomy = args.taxonomy
    ojo_skill_file_name = args.ojo_skill_fn

    skill_mapper = SkillMapper(
        taxonomy=taxonomy,
        skill_name_col="preferredLabel",
        skill_desc_col="description",
        bert_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        multi_process=True,
        batch_size=32,
        ojo_skills_file_name=config["ojo_skills_ner_path"],
    )

    ##TO DO: Modify class to take list instead
    #esco_data = get_s3_data_paths(
    #    get_s3_resource(), bucket_name, "escoe_extension/inputs/data/esco", "*.csv"
    #)
    #esco_dfs = {
    #    esco_df.split("/")[-1].split("_")[0]: load_s3_data(
    #        get_s3_resource(), bucket_name, esco_df
    #    )
    #    for esco_df in esco_data
    #}
    #all_skills = (
    #    list(esco_dfs["skillGroups"]["preferredLabel"])
    #    + list(esco_dfs["skills"]["preferredLabel"])
    #    + list(esco_dfs["skillsHierarchy"]["Level 1 preferred term"])
    #    + list(esco_dfs["skillsHierarchy"]["Level 2 preferred term"])
    #    + list(esco_dfs["skillsHierarchy"]["Level 3 preferred term"])
    #)
    #taxonomy_skill_list = [i for i in list(set(all_skills)) if type(i) != float]

    skills_to_taxonomy = skill_mapper.map_skills(
        taxonomy, taxonomy_skill_list, ojo_skill_file_name
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
