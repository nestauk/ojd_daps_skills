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
from ojd_daps_skills.getters.download_public_data import download
from ojd_daps_skills import logger, PROJECT_DIR, PUBLIC_DATA_FOLDER_NAME

import yaml
import os
import logging
from typing import List, Union, Optional
from ojd_daps_skills.utils.text_cleaning import short_hash


class ExtractSkills(object):
    """Class to extract skills from job adverts and map them to a skills taxonomy.

    :param config_path: The file name for the config file to be used, defaults to "extract_skills_toy"
    :type config_path: str

    :param local: Whether you want to load data from local files (True, if not found they will be downloaded from a public source) or via Nesta's private s3 bucket (False, needs access), defaults to True
    :type local: bool

    :param verbose: Whether to limit the number of logging messages (True) or not (False, good for debugging), defaults to True
    :type verbose: bool

    :param multi_process: Whether to use multiprocessing (True) or not (False), defaults to False
    :type multi_process: bool

    """

    def __init__(
        self,
        config_name="extract_skills_toy",
        local=True,
        verbose=True,
        multi_process=False,
    ):
        # Set variables from the config file
        config_path = os.path.join(
            PROJECT_DIR, "ojd_daps_skills/config/", config_name + ".yaml"
        )
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.local = local
        self.verbose = verbose
        self.multi_process = multi_process
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)
        if self.local:
            self.s3 = False
            self.base_path = PUBLIC_DATA_FOLDER_NAME + "/"
            if not os.path.exists(os.path.join(PROJECT_DIR, PUBLIC_DATA_FOLDER_NAME)):
                logger.warning(
                    "Neccessary files are not downloaded. Downloading ~1GB of neccessary files."
                )
                download()
        else:
            self.base_path = "escoe_extension/"
            self.s3 = True
            pass

        self.taxonomy_name = self.config["taxonomy_name"]
        self.taxonomy_path = os.path.join(self.base_path, self.config["taxonomy_path"])
        self.clean_job_ads = self.config["clean_job_ads"]
        self.min_multiskill_length = self.config["min_multiskill_length"]
        self.taxonomy_embedding_file_name = self.config.get(
            "taxonomy_embedding_file_name"
        )
        if self.taxonomy_embedding_file_name:
            self.taxonomy_embedding_file_name = os.path.join(
                self.base_path, self.taxonomy_embedding_file_name
            )
        self.prev_skill_matches_file_name = self.config.get(
            "prev_skill_matches_file_name"
        )
        if self.prev_skill_matches_file_name:
            self.prev_skill_matches_file_name = os.path.join(
                self.base_path, self.prev_skill_matches_file_name
            )
        self.hard_labelled_skills_file_name = self.config.get(
            "hard_labelled_skills_file_name"
        )
        if self.hard_labelled_skills_file_name:
            self.hard_labelled_skills_file_name = os.path.join(
                self.base_path, self.hard_labelled_skills_file_name
            )
        self.hier_name_mapper_file_name = self.config.get("hier_name_mapper_file_name")
        if self.hier_name_mapper_file_name:
            self.hier_name_mapper_file_name = os.path.join(
                self.base_path, self.hier_name_mapper_file_name
            )

        if self.local:
            self.ner_model_path = os.path.join(
                PROJECT_DIR, self.base_path, self.config["ner_model_path"]
            )
        else:
            self.ner_model_path = os.path.join(
                self.base_path, self.config["ner_model_path"]
            )

    def load(
        self,
        taxonomy_embedding_file_name: Optional[str] = None,
        prev_skill_matches_file_name: Optional[str] = None,
        hard_labelled_skills_name: Optional[str] = None,
        hier_name_mapper_file_name: Optional[str] = None,
    ):
        """Loads necessary datasets (formatted taxonomy, hard labelled skills, previously matched skills,
        taxonomy embeddings), JobNER skills extraction class and SkillMapper skill mapper class.

        :param taxonomy_embedding_file_name: The relative path to a taxonomy embedding file if it exists. If left unset the embeddings will be generated when the code is run. Defaults to None.
        :type taxonomy_embedding_file_name: str, optional

        :param prev_skill_matches_file_name: The relative path to a previous skill matches file if it exists. Defaults to None.
        :type prev_skill_matches_file_name: str, optional

        :param hard_labelled_skills_name: The relative path to a hard labelled skills file if it exists. Defaults to None.
        :type hard_labelled_skills_name: str, optional

        :param hier_name_mapper_file_name: The relative path to a hierarchy name mapper file if it exists. Defaults to None.
        :type hier_name_mapper_file_name: str, optional

        """

        if (not taxonomy_embedding_file_name) and (self.taxonomy_embedding_file_name):
            taxonomy_embedding_file_name = self.taxonomy_embedding_file_name
        if (not prev_skill_matches_file_name) and (self.prev_skill_matches_file_name):
            prev_skill_matches_file_name = self.prev_skill_matches_file_name
        if (not hard_labelled_skills_name) and (self.hard_labelled_skills_file_name):
            hard_labelled_skills_name = self.hard_labelled_skills_file_name
        if (not hier_name_mapper_file_name) and (self.hier_name_mapper_file_name):
            hier_name_mapper_file_name = self.hier_name_mapper_file_name

        self.job_ner = JobNER()

        self.nlp = self.job_ner.load_model(self.ner_model_path, s3_download=self.s3)

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
            multi_process=self.multi_process,
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
                f"Loading pre-defined or previously found skill mappings from {prev_skill_matches_file_name}"
            )
            self.prev_skill_matches = self.skill_mapper.load_ojo_esco_mapper(
                self.prev_skill_matches_file_name, s3=self.s3
            )
            # self.prev_skill_matches = {1654958883999821: {'ojo_skill': 'maths skills', 'match_skill': 'communicate with others', 'match_score': 0.3333333333333333, 'match_type': 'most_common_level_1', 'match_id': 'S1.1'}}
        else:
            self.prev_skill_matches = None

        if hard_labelled_skills_name:
            logger.info(
                f"Loading hard coded skill mappings for top skills in {hard_labelled_skills_name}"
            )
            self.hard_coded_skills = self.skill_mapper.load_ojo_esco_mapper(
                self.hard_labelled_skills_file_name, s3=self.s3
            )
            # self.hard_coded_skills = {1654958883999821: {'ojo_skill': 'maths skills', 'match_skill': 'communicate with others', 'match_id': 'S1.1'}}
        else:
            self.hard_coded_skills = None

    def format_skills(self, skills: List[str]) -> List[dict]:
        """Format list of skills from a single job advert to be in the format needed for mapping to a taxonomy. Also applies the
        multiskill splitting to any skills predicted to be multiskills.

        :param skills: A list of skills/multiskills from the job advert or a single skill
        :type skills: str or list of strings

        :return: The skills arranged into the format [{"SKILL": [...], "MULTISKILL": [...], "EXPERIENCE": []}]
        :rtype: a list of length 1 containing a dictionary

        """

        if isinstance(skills, str):
            skills = [skills]

        ms_classifier = self.job_ner.ms_classifier

        all_split_skills = []
        multiskills = []
        for skill in skills:
            if ms_classifier.predict(skill) == 1:
                split_list = split_multiskill(
                    skill, min_length=self.min_multiskill_length
                )
                if split_list:
                    all_split_skills.extend(split_list)
                else:
                    multiskills.append(skill)
            else:
                all_split_skills.append(skill)

        skill_dict = {}
        skill_dict["SKILL"] = all_split_skills
        skill_dict["MULTISKILL"] = multiskills
        skill_dict["EXPERIENCE"] = []

        logger.info(
            f"reformatted list of skills to map to '{self.taxonomy_name}' taxonomy"
        )

        return [skill_dict]

    def get_skills(self, job_adverts: Union[str, List[str]]):
        """Predict skill/multiskill/experience entities using the NER model in inputted job adverts.
        Multiskill entities will be split up and converted into individual skill entities where possible.

        :param job_adverts: The text of a job advert or a list of job adverts texts
        :type job_adverts: str or list of strings

        :return: A list of entities extracted from each job advert in the form of dictionaries {"SKILL": ["Microsoft Excel"], "MULTISKILL": [], "EXPERIENCE": []}
        :rtype: list, the length is equal to the number of job adverts inputted

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

    def map_skills(self, predicted_skills: Union[List[dict], List[str]]):
        """Map skills from job advert(s) to a skills taxonomy. If predicted_skills is a list of skills, it will be formatted accordingly to
        be mapped to a skills taxonomy. All multiskill entities will be mapped in the same way as skill entities are.

        :param predicted_skills: A list of skill entities either in the form of a list of strings (assumed to be from the same job advert) or a list of the dictionaries outputted from the get_skills function.
        :type predicted_skills: list of strings or a list of dicts

        :return: A list of dictionaries for each job advert containing the skill and experience entities, and for every skill entity where it maps to in the taxonomy. Multi skill entities are treated as skill entities, and the output combines them together as one. Each dictionary is in the format {'SKILL': [(skill_entity,(taxonomy_skill_name, taxonomy_skill_id)), ...]}, 'EXPERIENCE': [...]]
        :rtype: list of dictionaries for each job advert.

        """
        if isinstance(predicted_skills[0], str):
            predicted_skills = self.format_skills(predicted_skills)

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
            self.skill_matches = self.skill_mapper.final_prediction(
                fully_mapped_skills,
                self.taxonomy_info.get("hier_name_mapper"),
                self.taxonomy_info.get("match_thresholds_dict"),
                self.taxonomy_info.get("num_hier_levels"),
            )

            if self.prev_skill_matches:
                # Append the pre-defined matches with the new matches
                self.skill_matches = self.skill_mapper.append_final_predictions(
                    self.skill_matches, self.prev_skill_matches
                )

            _, job_skills_matched = self.skill_mapper.link_skill_hash_to_job_id(
                job_skills, self.skill_matches
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

        if self.hard_coded_skills:
            for formatted_skill in job_skills_matched_formatted:
                if "SKILL" in formatted_skill.keys():
                    extracted_skills = formatted_skill["SKILL"]
                    skills_to_hard_code = []
                    for skill in extracted_skills:
                        skill_hash_str = str(short_hash(skill[0]))
                        hard_coded_skill = self.hard_coded_skills.get(skill_hash_str)
                        if hard_coded_skill:
                            skills_to_hard_code.append(
                                (
                                    skill[0],
                                    (
                                        hard_coded_skill["match_skill"],
                                        hard_coded_skill["match_id"],
                                    ),
                                )
                            )
                        else:
                            skills_to_hard_code.append(skill)
                    formatted_skill["SKILL"] = skills_to_hard_code

        return job_skills_matched_formatted

    def extract_skills(
        self, job_adverts_skills: Union[str, List[str]], format_skills=False
    ):
        """Extract skills from job adverts using a trained NER model and map them to a taxonomy - combines both get_skills and extract_skills. Experiences will also be extracted, but not mapped to a taxonomy. It can also take as input a list of
        skills and map them to a taxonomy if format_skills is set to True.

        :param job_adverts_skills: The text of a job advert, a list of job adverts texts, or a list of skills (if format_skills=True)
        :type job_adverts_skills: str or list of strings

        :param format_skills: If the input is a list of skills (rather than job adverts) then this needs to be set to True in order to format them correctly, default to False.
        :type format_skills: bool

        :return: A list of dictionaries for each job advert containing the skill and experience entities, and for every skill entity where it maps to in the taxonomy. The output combines both multiskill and skill entities together in the "SKILL" key. Each dictionary is in the format {'SKILL': [(skill_entity,(taxonomy_skill_name, taxonomy_skill_id)), ...]}, 'EXPERIENCE': [...]]
        :rtype: list of dictionaries for each job advert.

        """
        if format_skills:
            skills = self.format_skills(job_adverts_skills)
            logger.info(
                f"formatted {len(job_adverts_skills)} skill(s) from skills list..."
            )
        else:
            skills = self.get_skills(job_adverts_skills)

        mapped_skills = self.map_skills(skills)

        return mapped_skills


if __name__ == "__main__":

    es = ExtractSkills(config_name="extract_skills_esco", local=True)

    es.load()

    job_adverts = [
        "You will need to have good communication and mathematics skills. You will have experience in the IT sector.",
        "You will need to have good excel and presenting skills. You need good excel software skills",
    ]

    skills_list = ["communication", "excel skills", "dancing", "singing"]
    # 2 steps
    predicted_skills = es.get_skills(job_adverts)
    job_skills_matched = es.map_skills(predicted_skills)

    # 1 step, list of skills
    job_skills_matched = es.map_skills(skills_list)

    # # 1 step - get then extract
    job_skills_matched_one_step = es.extract_skills(job_adverts)
    # # 1 step - format then extract
    job_skills_list = es.extract_skills(skills_list, format_skills=True)
