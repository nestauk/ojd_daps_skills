"""
Extract skills from a list of job adverts and match them to a chosen taxonomy
"""
from ojd_daps_skills.pipeline.skill_ner.ner_spacy import JobNER
from ojd_daps_skills.utils.text_cleaning import clean_text
from ojd_daps_skills.pipeline.skill_ner.multiskill_utils import split_multiskill
from ojd_daps_skills.pipeline.skill_ner_mapping.skill_ner_mapper import SkillMapper
from ojd_daps_skills.pipeline.extract_skills.extract_skills_utils import (
    load_toy_taxonomy,
)
from ojd_daps_skills.getters.data_getters import load_file
from ojd_daps_skills import logger, PROJECT_DIR

import yaml
import os
import logging
import pickle
from typing import List

from ojd_daps_skills.pipeline.skill_ner.multiskill_utils import MultiskillClassifier


class ExtractSkills(object):
    """
    Class to extract skills from job adverts and map them to a skills taxonomy.
    Attributes
    ----------
    config_path (str): the config path for a default setting
    s3 (bool): whether you want to load/save data from this repos from s3 bucket (True, needs access) or locally (False)
    ----------
    Methods
    ----------
    load(taxonomy_embedding_file_name, prev_skill_matches_file_name, hier_name_mapper_file_name)
        loads all the neccessary data and models for this class
    get_skills(job_adverts)
        For an inputted list of job adverts, or a single job advert text, predict skill/experience entities
    format_skills(skills)
        If input is list of skills, format list to be the output of get_skills - a list of dict to map
        each entity onto a skill/skill group from a taxonomy
    map_skills(predicted_skills)
        For a list of predicted skills (the output of get_skills - a list of dicts), map each entity
        onto a skill/skill group from a taxonomy
    extract_skills(job_adverts, map_to_tax=True)
        Does both get_skills and extract_skills if map_to_tax=True, otherwise just does get_skills
    """

    def __init__(self, config_name="extract_skills_toy", s3=True, verbose=True):
        # Set variables from the config file
        config_path = os.path.join(
            PROJECT_DIR, "ojd_daps_skills/config/", config_name + ".yaml"
        )
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.s3 = s3
        self.verbose = verbose
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)
        if self.s3:
            pass
        else:
            if os.path.exists(str(PROJECT_DIR) + "/escoe_extension"):
                logger.info("data and embeddings already downloaded locally.")
            else:
                cmd = f"aws s3 sync s3://open-jobs-indicators/escoe_extension {str(PROJECT_DIR) + '/escoe_extension'}"
                os.system(cmd)
                logger.info("downloaded data and embeddings locally.")

        self.ner_model_path = self.config["ner_model_path"]
        self.taxonomy_name = self.config["taxonomy_name"]
        self.taxonomy_path = self.config["taxonomy_path"]
        self.clean_job_ads = self.config["clean_job_ads"]
        self.min_multiskill_length = self.config["min_multiskill_length"]
        self.taxonomy_embedding_file_name = self.config.get(
            "taxonomy_embedding_file_name"
        )
        self.prev_skill_matches_file_name = self.config.get(
            "prev_skill_matches_file_name"
        )
        self.hier_name_mapper_file_name = self.config.get("hier_name_mapper_file_name")

    def load(
        self,
        taxonomy_embedding_file_name=None,
        prev_skill_matches_file_name=None,
        hier_name_mapper_file_name=None,
    ):
        """
        Loads necessary datasets, JobNER skills extraction class and SkillMapper skill mapper class
        """

        if (not taxonomy_embedding_file_name) and (self.taxonomy_embedding_file_name):
            taxonomy_embedding_file_name = self.taxonomy_embedding_file_name
        if (not prev_skill_matches_file_name) and (self.prev_skill_matches_file_name):
            prev_skill_matches_file_name = self.prev_skill_matches_file_name
        if (not hier_name_mapper_file_name) and (self.hier_name_mapper_file_name):
            hier_name_mapper_file_name = self.hier_name_mapper_file_name

        self.job_ner = JobNER()

        if self.s3:
            self.nlp = self.job_ner.load_model(self.ner_model_path, s3_download=self.s3)
        else:
            self.nlp = self.job_ner.load_model(
                str(PROJECT_DIR) + "/escoe_extension/" + self.config["ner_model_path"],
                s3_download=self.s3,
            )

        self.labels = self.nlp.get_pipe("ner").labels + ("MULTISKILL",)

        logger.info(f"Loading '{self.taxonomy_name}' taxonomy information")
        if self.taxonomy_name == "toy":
            self.taxonomy_skills = load_toy_taxonomy()
        else:
            if hier_name_mapper_file_name:
                self.hier_name_mapper = load_file(
                    hier_name_mapper_file_name, s3=self.s3
                )
            else:
                self.hier_name_mapper = {}
            self.config["hier_name_mapper"] = self.hier_name_mapper

        taxonomy_info_names = [
            "num_hier_levels",
            "skill_type_dict",
            "match_thresholds_dict",
            "hier_name_mapper",
            "skill_name_col",
            "skill_id_col",
            "skill_hier_info_col",
            "skill_type_col",
        ]
        self.taxonomy_info = {
            name: self.config.get(name) for name in taxonomy_info_names
        }

        self.skill_mapper = SkillMapper(
            taxonomy=self.taxonomy_name,
            skill_name_col=self.taxonomy_info.get("skill_name_col"),
            skill_id_col=self.taxonomy_info.get("skill_id_col"),
            skill_hier_info_col=self.taxonomy_info.get("skill_hier_info_col"),
            skill_type_col=self.taxonomy_info.get("skill_type_col"),
            verbose=self.verbose,
        )

        if self.taxonomy_name != "toy":
            self.taxonomy_skills = self.skill_mapper.load_taxonomy_skills(
                self.taxonomy_path, s3=self.s3
            )
            self.taxonomy_skills = self.skill_mapper.preprocess_taxonomy_skills(
                self.taxonomy_skills
            )

        if taxonomy_embedding_file_name:
            logger.info(
                f"Loading taxonomy embeddings from {taxonomy_embedding_file_name}"
            )
            _ = self.skill_mapper.load_taxonomy_embeddings(
                taxonomy_embedding_file_name, s3=self.s3
            )
            self.taxonomy_skills_embeddings_loaded = True
        else:
            self.taxonomy_skills_embeddings_loaded = False

        if prev_skill_matches_file_name:
            logger.info(
                f"Loading previously found skill mappings from {prev_skill_matches_file_name}"
            )
            self.prev_skill_matches = self.skill_mapper.load_ojo_esco_mapper(
                self.prev_skill_matches_file_name, s3=self.s3
            )
            # self.prev_skill_matches = {1654958883999821: {'ojo_skill': 'maths skills', 'match_skill': 'communicate with others', 'match_score': 0.3333333333333333, 'match_type': 'most_common_level_1', 'match_id': 'S1.1'}}
        else:
            self.prev_skill_matches = None

    def format_skills(self, skills: List[str]) -> List[dict]:
        """
        If input is list of skills strings, format list to map to taxonomy
        """

        if isinstance(skills, str):
            skills = [skills]

        if self.s3:
            ms_classifier = pickle.load(
                open(self.ner_model_path + "ms_classifier.pkl", "rb")
            )
        else:
            ms_classifier = pickle.load(
                open(
                    str(PROJECT_DIR)
                    + "/escoe_extension/"
                    + self.config["ner_model_path"]
                    + "ms_classifier.pkl",
                    "rb",
                )
            )

        extracted_skills = []
        for skill in skills:
            skill_dict = {}
            if ms_classifier.predict(skill) == 1:
                split_list = split_multiskill(
                    skill, min_length=self.min_multiskill_length
                )
                if split_list:
                    for split_entity in split_list:
                        skill_dict["SKILL"] = [split_entity]
                else:
                    skill_dict["SKILL"] = [skill]
            else:
                skill_dict["SKILL"] = [skill]

            skill_dict["MULTISKILL"] = []
            skill_dict["EXPERIENCE"] = []

            extracted_skills.append(skill_dict)
        logger.info(f"reformatted list of skills to map to '{self.taxonomy}' taxonomy")

        return extracted_skills

    def get_skills(self, job_adverts):
        """
        Extract skills using the NER model from a single or a list of job adverts
        """

        if isinstance(job_adverts, str):
            job_adverts = [job_adverts]

        predicted_skills = []
        for job_advert in job_adverts:
            if self.clean_job_ads:
                job_advert = clean_text(job_advert)
            skill_ents = self.job_ner.predict(job_advert)
            skills = {label: [] for label in self.labels}
            for ent in skill_ents:
                label = ent["label"]
                ent_text = job_advert[ent["start"] : ent["end"]]
                if label == "MULTISKILL":
                    split_list = split_multiskill(
                        ent_text, min_length=self.min_multiskill_length
                    )
                    if split_list:
                        # If we can split up the multiskill into individual skills
                        for split_entity in split_list:
                            skills["SKILL"].append(split_entity)
                    else:
                        # We havent split up the multiskill, just add it all in
                        skills[label].append(ent_text)
                else:
                    skills[label].append(ent_text)
            predicted_skills.append(skills)
        return predicted_skills

    def map_skills(self, predicted_skills):
        """
        Maps a list of skills to a skills taxonomy
        """

        skills = {"predictions": {i: s for i, s in enumerate(predicted_skills)}}
        job_skills, skill_hashes = self.skill_mapper.preprocess_job_skills(skills)
        if len(skill_hashes) != 0:
            logger.info(
                f"Mapping {len(skill_hashes)} skills to the '{self.taxonomy_name}' taxonomy"
            )
            if self.prev_skill_matches:
                orig_num = len(skill_hashes)
                skill_hashes = self.skill_mapper.filter_skill_hash(
                    skill_hashes, self.prev_skill_matches
                )
                logger.info(f"{orig_num - len(skill_hashes)} mappings previously found")

            if not self.taxonomy_skills_embeddings_loaded:
                # If we didn't already load the embeddings, then calculate them
                self.skill_mapper.embed_taxonomy_skills(self.taxonomy_skills)

            fully_mapped_skills = self.skill_mapper.map_skills(
                self.taxonomy_skills,
                skill_hashes,
                self.taxonomy_info.get("num_hier_levels"),
                self.taxonomy_info.get("skill_type_dict"),
            )
            skill_matches = self.skill_mapper.final_prediction(
                fully_mapped_skills,
                self.taxonomy_info.get("hier_name_mapper"),
                self.taxonomy_info.get("match_thresholds_dict"),
                self.taxonomy_info.get("num_hier_levels"),
            )

            if self.prev_skill_matches:
                # Append the pre-defined matches with the new matches
                skill_matches = self.skill_mapper.append_final_predictions(
                    skill_matches, self.prev_skill_matches
                )

            _, job_skills_matched = self.skill_mapper.link_skill_hash_to_job_id(
                job_skills, skill_matches
            )

            job_skills_matched_formatted = []
            for ix, _ in skills["predictions"].items():
                # Go through input dict, try to find matches, but
                # if there were no skills then this job key won't be in
                # job_skills_matched.
                job_skills_info = job_skills_matched.get(ix)
                if job_skills_info:
                    skill_list = list(
                        zip(
                            job_skills_info["clean_skills"],
                            [
                                (j["match_skill"], j["match_id"])
                                for j in job_skills_info["skill_to_taxonomy"]
                            ],
                        )
                    )
                    try:
                        experience_list = predicted_skills[ix]["EXPERIENCE"]
                    except:
                        experience_list = []

                    job_skills_matched_formatted.append(
                        {
                            k: v
                            for k, v in [
                                ("SKILL", skill_list),
                                ("EXPERIENCE", experience_list),
                            ]
                            if v
                        }
                    )
                else:
                    # This means we keep the number of job adverts in the input list
                    # the same as the number in the output list
                    job_skills_matched_formatted.append({})
        else:
            job_skills_matched_formatted = [{} for _ in range(len(predicted_skills))]

        return job_skills_matched_formatted

    def extract_skills(self, job_adverts, map_to_tax=True):
        """
        Extract skills using the NER model from a single or a list of job adverts
        and if map_to_tax==True then also map them to the taxonomy
        """
        skills = self.get_skills(job_adverts)
        if map_to_tax:
            mapped_skills = self.map_skills(skills)
            return mapped_skills
        else:
            return skills


if __name__ == "__main__":

    es = ExtractSkills(config_name="extract_skills_esco", s3=True)

    es.load()

    job_adverts = [
        "You will need to have good communication and mathematics skills. You will have experience in the IT sector.",
        "You will need to have good excel and presenting skills. You need good excel software skills",
    ]

    # 2 steps
    predicted_skills = es.get_skills(job_adverts)
    job_skills_matched = es.map_skills(predicted_skills)

    # # 1 step
    # job_skills_matched = es.extract_skills(job_adverts, map_to_tax=True)
    # # 1 step
    # predicted_skills = es.extract_skills(job_adverts, map_to_tax=False)
