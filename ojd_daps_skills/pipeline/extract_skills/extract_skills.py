from ojd_daps_skills.pipeline.skill_ner.ner_spacy import JobNER
from ojd_daps_skills.utils.text_cleaning import clean_text
from ojd_daps_skills.pipeline.skill_ner.multiskill_utils import split_multiskill
from ojd_daps_skills.pipeline.skill_ner_mapping.skill_ner_mapper import SkillMapper
from ojd_daps_skills.pipeline.extract_skills.extract_skills_utils import (
    load_toy_taxonomy,
)


class ExtractSkills(object):
    def __init__(
        self,
        ner_model_path="outputs/models/ner_model/20220729/",
        s3=True,
        clean_job_ads=True,
        min_multiskill_length=75,
        taxonomy="toy",
    ):
        self.ner_model_path = ner_model_path
        self.s3 = s3
        self.clean_job_ads = clean_job_ads
        self.min_multiskill_length = min_multiskill_length
        self.taxonomy = taxonomy

    def load_things(self):
        """
        Try to load as much as a one off as we can
        """

        self.job_ner = JobNER()
        self.nlp = self.job_ner.load_model(self.ner_model_path, s3_download=self.s3)
        self.labels = self.nlp.get_pipe("ner").labels + ("MULTISKILL",)

        # Load taxonomy
        if self.taxonomy == "toy":
            self.taxonomy_skills, self.taxonomy_info = load_toy_taxonomy()
        # elif taxonomy=='esco':
        #     self.taxonomy_skills

        self.skill_mapper = SkillMapper(
            skill_name_col=self.taxonomy_info["skill_name_col"],
            skill_id_col=self.taxonomy_info["skill_id_col"],
            skill_hier_info_col=self.taxonomy_info["skill_hier_info_col"],
            skill_type_col=self.taxonomy_info["skill_type_col"],
        )

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

    def map_skills(self, skills_list):
        """
        Map a list of skills to ESCO
        """
        skills = {"predictions": {i: s for i, s in enumerate(skills_list)}}
        clean_ojo_skills, skill_hashes = self.skill_mapper.preprocess_job_skills(skills)

        self.skill_mapper.embed_taxonomy_skills(
            self.taxonomy_skills, "notneeded", save=False
        )
        skills_to_taxonomy = self.skill_mapper.map_skills(
            self.taxonomy_skills,
            skill_hashes,
            self.taxonomy_info["num_hier_levels"],
            self.taxonomy_info["skill_type_dict"],
        )
        final_match = self.skill_mapper.final_prediction(
            skills_to_taxonomy,
            self.taxonomy_info["hier_name_mapper"],
            self.taxonomy_info["match_thresholds_dict"],
            self.taxonomy_info["num_hier_levels"],
        )

        _, final_ojo_skills = self.skill_mapper.link_skill_hash_to_job_id(
            clean_ojo_skills, final_match
        )

        f_final_ojo_skills = []
        for ix, job_skills in final_ojo_skills.items():
            f_final_ojo_skills.append(
                {
                    "SKILL": list(
                        zip(
                            job_skills["clean_skills"],
                            [
                                (j["match_skill"], j["match_id"])
                                for j in job_skills["skill_to_taxonomy"]
                            ],
                        )
                    ),
                    "EXPERIENCE": skills_list[ix]["EXPERIENCE"],
                }
            )

        return f_final_ojo_skills

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

    es = ExtractSkills(
        ner_model_path="outputs/models/ner_model/20220729/", s3=True, taxonomy="toy"
    )
    es.load_things()

    job_adverts = [
        "The job involves communication and maths skills",
        "The job involves excel and presenting skills. You need good excel skills",
    ]

    # 2 steps
    predicted_skills = es.get_ner_skills(job_adverts)
    final_ojo_skills = es.map_skills(predicted_skills)

    # # 1 step
    # final_ojo_skills = es.extract_skills(job_adverts, map_to_tax=True)
    # # 1 step
    # predicted_skills = es.extract_skills(job_adverts, map_to_tax=False)
