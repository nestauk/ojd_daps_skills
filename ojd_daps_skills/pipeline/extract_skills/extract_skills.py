"""
Extract skills from a list of job adverts and match them to a chosen taxonomy
"""

from ojd_daps_skills.pipeline.skill_ner.ner_spacy import JobNER
from ojd_daps_skills.utils.text_cleaning import clean_text
from ojd_daps_skills.pipeline.skill_ner.multiskill_utils import split_multiskill
from ojd_daps_skills.pipeline.skill_ner_mapping.skill_ner_mapper import SkillMapper
from ojd_daps_skills.pipeline.extract_skills.extract_skills_utils import (
    load_toy_taxonomy,
    load_esco_taxonomy_info,
)
from ojd_daps_skills.getters.data_getters import load_file
from ojd_daps_skills import logger


class ExtractSkills(object):
    """
    Class to extract skills from job adverts and map them to a skills taxonomy.
    Attributes
    ----------
    ner_model_path (str): the path to the trained NER model (either s3 or local)
    s3 (bool): whether you want to load/save data from this repos s3 bucket (True, needs access) or locally (False)
    clean_job_ads (bool): whether you want to clean the job advert text before extracting skills (True) or not (False)
    min_multiskill_length (int): the maximum entity length for which a multiskill entity will be split into skills
    taxonomy_path (str): The filepath to the formatted taxonomy, can also set to "toy" to use a toy taxonomy
    ----------
    Methods
    ----------
    load_things(taxonomy_embedding_file_name, prev_skill_matches_filename, hier_name_mapper_file_name)
        loads all the neccessary data and models for this class
    get_ner_skills(job_adverts)
        For an inputted list of job adverts, or a single job advert text, predict skill/experience entities
    map_skills(predicted_skills)
        For a list of predicted skills (the output of get_ner_skills - a list of dicts), map each entity
        onto a skill/skill group from a taxonomy
    extract_skills(job_adverts, map_to_tax=True)
        Does both get_ner_skills and extract_skills if map_to_tax=True, otherwise just does get_ner_skills
    """

    def __init__(
        self,
        ner_model_path="outputs/models/ner_model/20220729/",
        s3=True,
        clean_job_ads=True,
        min_multiskill_length=75,
        taxonomy_path="toy",
    ):
        self.ner_model_path = ner_model_path
        self.s3 = s3
        self.clean_job_ads = clean_job_ads
        self.min_multiskill_length = min_multiskill_length
        self.taxonomy_path = taxonomy_path

    def load_things(
        self,
        taxonomy_embedding_file_name=None,
        prev_skill_matches_filename=None,
        hier_name_mapper_file_name=None,
    ):
        """
        Try to load as much as a one off as we can
        """

        self.job_ner = JobNER()
        self.nlp = self.job_ner.load_model(self.ner_model_path, s3_download=self.s3)
        self.labels = self.nlp.get_pipe("ner").labels + ("MULTISKILL",)

        # Load taxonomy
        logger.info(f"Loading taxonomy information from {self.taxonomy_path}")
        if self.taxonomy_path == "toy":
            self.taxonomy_skills, self.taxonomy_info = load_toy_taxonomy()

            self.skill_mapper = SkillMapper(
                skill_name_col=self.taxonomy_info["skill_name_col"],
                skill_id_col=self.taxonomy_info["skill_id_col"],
                skill_hier_info_col=self.taxonomy_info["skill_hier_info_col"],
                skill_type_col=self.taxonomy_info["skill_type_col"],
            )
        else:
            self.taxonomy_info = load_esco_taxonomy_info()
            if hier_name_mapper_file_name:
                hier_name_mapper = load_file(hier_name_mapper_file_name, s3=self.s3)
            else:
                hier_name_mapper = {}
            self.taxonomy_info["hier_name_mapper"] = hier_name_mapper
            self.skill_mapper = SkillMapper(
                skill_name_col=self.taxonomy_info["skill_name_col"],
                skill_id_col=self.taxonomy_info["skill_id_col"],
                skill_hier_info_col=self.taxonomy_info["skill_hier_info_col"],
                skill_type_col=self.taxonomy_info["skill_type_col"],
            )

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

        if prev_skill_matches_filename:
            logger.info(
                f"Loading previously found skill mappings from {prev_skill_matches_filename}"
            )
            self.prev_skill_matches = load_ojo_esco_mapper(
                self.skill_mapper.prev_skill_matches_filename, s3=self.s3
            )
            # self.prev_skill_matches = {1654958883999821: {'ojo_skill': 'maths skills', 'match_skill': 'communicate with others', 'match_score': 0.3333333333333333, 'match_type': 'most_common_level_1', 'match_id': 'S1.1'}}
        else:
            self.prev_skill_matches = None

    def get_ner_skills(self, job_adverts):
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
        Map a list of skills to ESCO
        """

        skills = {"predictions": {i: s for i, s in enumerate(predicted_skills)}}
        job_skills, skill_hashes = self.skill_mapper.preprocess_job_skills(skills)
        logger.info(f"Mapping {len(skill_hashes)} skills to the taxonomy")
        if self.prev_skill_matches:
            orig_num = len(skill_hashes)
            skill_hashes = self.skill_mapper.filter_skill_hash(
                skill_hashes, self.prev_skill_matches
            )
            logger.info(f"{orig_num - len(skill_hashes)} mappings previously found")

        if not self.taxonomy_skills_embeddings_loaded:
            # If we didn't already load the embeddings, then calculate them
            self.skill_mapper.embed_taxonomy_skills(
                self.taxonomy_skills, "notneeded", save=False
            )

        fully_mapped_skills = self.skill_mapper.map_skills(
            self.taxonomy_skills,
            skill_hashes,
            self.taxonomy_info["num_hier_levels"],
            self.taxonomy_info["skill_type_dict"],
        )
        skill_matches = self.skill_mapper.final_prediction(
            fully_mapped_skills,
            self.taxonomy_info.get("hier_name_mapper"),
            self.taxonomy_info["match_thresholds_dict"],
            self.taxonomy_info["num_hier_levels"],
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
        for ix, job_skills_info in job_skills_matched.items():
            skill_list = list(
                zip(
                    job_skills_info["clean_skills"],
                    [
                        (j["match_skill"], j["match_id"])
                        for j in job_skills_info["skill_to_taxonomy"]
                    ],
                )
            )
            experience_list = predicted_skills[ix]["EXPERIENCE"]

            job_skills_matched_formatted.append(
                {
                    k: v
                    for k, v in [("SKILL", skill_list), ("EXPERIENCE", experience_list)]
                    if v
                }
            )

        return job_skills_matched_formatted

    def extract_skills(self, job_adverts, map_to_tax=True):
        """
        Extract skills using the NER model from a single or a list of job adverts
        and if map_to_tax==True then also map them to the taxonomy
        """
        skills = self.get_ner_skills(job_adverts)
        if map_to_tax:
            mapped_skills = self.map_skills(skills)
            return mapped_skills
        else:
            return skills


if __name__ == "__main__":

    ner_model_path = "outputs/models/ner_model/20220729/"
    s3 = True
    taxonomy_path = "toy"
    clean_job_ads = True
    min_multiskill_length = 75

    taxonomy_embedding_file_name = None
    prev_skill_matches_filename = None
    hier_name_mapper_file_name = (
        "escoe_extension/outputs/data/skill_ner_mapping/esco_hier_mapper.json"
    )

    es = ExtractSkills(
        ner_model_path=ner_model_path,
        s3=s3,
        taxonomy_path=taxonomy_path,
        clean_job_ads=clean_job_ads,
        min_multiskill_length=min_multiskill_length,
    )

    es.load_things(
        taxonomy_embedding_file_name=taxonomy_embedding_file_name,
        prev_skill_matches_filename=prev_skill_matches_filename,
        hier_name_mapper_file_name=hier_name_mapper_file_name,
    )

    job_adverts = [
        "The job involves communication and maths skills",
        "The job involves excel and presenting skills. You need good excel skills",
    ]

    # 2 steps
    predicted_skills = es.get_ner_skills(job_adverts)
    job_skills_matched = es.map_skills(predicted_skills)

    # # 1 step
    # job_skills_matched = es.extract_skills(job_adverts, map_to_tax=True)
    # # 1 step
    # predicted_skills = es.extract_skills(job_adverts, map_to_tax=False)
