"""
This script is a one off script to calculate the ESCO embeddings
this is useful for extract_skills.py so we don't recalculate it everytime
"""

import os
import yaml

from ojd_daps_skills.pipeline.skill_ner_mapping.skill_ner_mapper import SkillMapper
from ojd_daps_skills import PROJECT_DIR

config_name = "extract_skills_esco"
esco_embeddings_output_file_name = (
    "escoe_extension/outputs/data/skill_ner_mapping/esco_embeddings.json"
)

if __name__ == "__main__":

    config_path = os.path.join(
        PROJECT_DIR, "ojd_daps_skills/config/", config_name + ".yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    skill_mapper = SkillMapper(
        skill_name_col=config["skill_name_col"],
        skill_id_col=config["skill_id_col"],
        skill_hier_info_col=config["skill_hier_info_col"],
        skill_type_col=config["skill_type_col"],
        verbose=True,
    )

    num_hier_levels = config["num_hier_levels"]
    skill_type_dict = config["skill_type_dict"]
    tax_input_file_name = config["taxonomy_path"]
    match_thresholds_dict = config["match_thresholds_dict"]
    hier_name_mapper_file_name = config["hier_name_mapper_file_name"]

    taxonomy_skills = skill_mapper.load_taxonomy_skills(tax_input_file_name, s3=True)
    taxonomy_skills = skill_mapper.preprocess_taxonomy_skills(taxonomy_skills)

    skill_mapper.embed_taxonomy_skills(
        taxonomy_skills,
    )
    skill_mapper.save_taxonomy_embeddings(esco_embeddings_output_file_name)
