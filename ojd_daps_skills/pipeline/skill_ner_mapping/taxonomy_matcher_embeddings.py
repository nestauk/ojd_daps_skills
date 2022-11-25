"""
This script is a one off script to calculate embeddings from a config file.

this is useful for extract_skills.py so we don't recalculate it everytime

python ojd_daps_skills/pipeline/skill_ner_mapping/taxonomy_matcher_embeddings.py --config_name CONFIG_NAME --embed_fn EMBEDDING_FILE_NAME
"""

import os
import yaml

from ojd_daps_skills.pipeline.skill_ner_mapping.skill_ner_mapper import SkillMapper
from ojd_daps_skills import PROJECT_DIR
from argparse import ArgumentParser

if __name__ == "__main__":

    embeddings_output_file_name = (
    "escoe_extension/outputs/data/skill_ner_mapping/"
    )

    parser = ArgumentParser()

    parser.add_argument(
        "--config_name",
        help="taxonomy config name",
    )

    parser.add_argument("--embed_fn", help="embedding file name")

    args = parser.parse_args()

    config_name = args.config_name
    embed_fn = args.embed_fn

    config_path = os.path.join(
        PROJECT_DIR, "ojd_daps_skills/config/", config_name + ".yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    skill_mapper = SkillMapper(
        taxonomy=config["taxonomy_name"],
        skill_name_col=config["skill_name_col"],
        skill_id_col=config["skill_id_col"],
        skill_hier_info_col=config["skill_hier_info_col"],
        skill_type_col=config["skill_type_col"],
        verbose=True,
    )

    num_hier_levels = config["num_hier_levels"]
    skill_type_dict = config["skill_type_dict"]
    tax_input_file_name = 'escoe_extension/' + config["taxonomy_path"]
    match_thresholds_dict = config["match_thresholds_dict"]
    hier_name_mapper_file_name = 'escoe_extension/' + config["hier_name_mapper_file_name"]

    taxonomy_skills = skill_mapper.load_taxonomy_skills(tax_input_file_name, s3=True)
    taxonomy_skills = skill_mapper.preprocess_taxonomy_skills(taxonomy_skills)

    skill_mapper.embed_taxonomy_skills(
        taxonomy_skills,
    )
    file_name = os.path.join(embeddings_output_file_name, embed_fn + '.json')
    skill_mapper.save_taxonomy_embeddings(file_name)
