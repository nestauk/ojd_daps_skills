"""
The taxonomy being mapped to in the script needs to be in a specific format.
There should be the 3 columns skill_name_col, skill_id_col, skill_type_col
with an optional 4th column (skill_hier_info_col).
### Example 1:
At the most basic level your taxonomy input could be:
"name" | "id" | "type"
---|---|---
"driving a car" | 123 | "skill"
"give presentations" | 333 | "skill"
"communicating well" | 456 | "skill"
...
with skill_type_dict = {'skill_types': ['skill']}.
Your output match for the OJO skill "communicate" might look like this:
{
'ojo_ner_skills': "communicate",
'top_5_tax_skills': [("communicating well", 456, 0.978), ("give presentations", 333, 0.762), ..]
}
- the closest skill to this ojo skill is "communicating well" which is code 456 and had a cosine distance of 0.978
### Example 2:
A more complicated example would have hierarchy levels given too
"name" | "id" | "type" | "hierarchy_levels"
---|---|---|---
"driving a car" | 123 | "skill" | ['A2.1']
"give presentations" | 333 | "skill" | ['A1.2']
"communicating well" | 456 | "skill"| ['A1.3']
...
with skill_type_dict = {'skill_types': ['skill']}.
This might give the result:
{
'ojo_ner_skills': "communicate",
'top_5_tax_skills': [("communicating well", 456, 0.978), ("give presentations", 333, 0.762), ..],
'high_tax_skills':  {'num_over_thresh': 2, 'most_common_level_0: ('A1', 1) , 'most_common_level_1': ('A1.3', 0.5)},
}
- 100% of the skills where the similarity is greater than the threshold are in the 'A1' skill level 0 group
- 50% of the skills where the similarity is greater than the threshold are in the 'A1.3' skill level 1 group
### Example 3:
And an even more complicated example would have skill level names given too (making use
of the 'type' column to differentiate them).
"name" | "id" | "type" | "hierarchy_levels"
---|---|---|---
"driving a car" | 123 | "skill" | ['A2.1']
"give presentations" | 333 | "skill" | ['A1.2']
"communicating well" | 456 | "skill"| ['A1.3']
"communication" | 'A1' | "level 1"| None
"driving" | 'A2' | "level 0"| None
"communicate verbally" | 'A1.3' | "level 1"| None
...
with skill_type_dict = {'skill_types': ['skill'], 'hier_types': ["level A", "level B"]} and num_hier_levels=2
This might give the result:
{
'ojo_ner_skills': "communicate",
'top_5_tax_skills': [("communicating well", 456, 0.978), ("give presentations", 333, 0.762), ..],
'high_tax_skills':  {'num_over_thresh': 2, 'most_common_level_0: ('A1', 1) , 'most_common_level_1': ('A1.3', 0.5)},
"top_'level 0'_tax_level": ('communication', 'A1', 0.998),
"top_'level 1'_tax_level": ('communicate verbally', 'A1.3', 0.98),
}
- the skill level 0 group 'communication' (code 'A1') is the closest to thie ojo skill with distance 0.998
- the skill level 1 group 'communicate verbally' (code 'A1.3') is the closest to thie ojo skill with distance 0.98
"""

import sys
sys.path.append("/Users/india.kerlenesta/Projects/ojd_daps_extension/ojd_daps_skills")

from ojd_daps_skills import config, bucket_name, PROJECT_DIR, logger
from ojd_daps_skills.getters.data_getters import (
    get_s3_resource,
    load_s3_data,
    save_to_s3,
    get_s3_data_paths,
    load_data,
    load_json_dict,
)
from ojd_daps_skills.pipeline.skill_ner_mapping.skill_ner_mapper_utils import (
    get_top_comparisons,
    get_most_common_code,
)
from ojd_daps_skills.utils.bert_vectorizer import BertVectorizer
from ojd_daps_skills.utils.text_cleaning import clean_text
from ojd_daps_skills.utils.logging import set_global_logging_level
from ojd_daps_skills.utils.text_cleaning import clean_text, short_hash

import logging
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer
import re
import time
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os

import ast
from ojd_daps_skills.pipeline.skill_ner.ner_spacy import JobNER
import boto3

S3 = get_s3_resource()

class SkillMapper:
    """
    Class to map extracted skills from NER model to a skills taxonomy.
    Attributes
    ----------
    skill_name_col (str): the taxonomy column name of the description of the skill/skill level
    skill_id_col (str): the taxonomy column name of the id for the skill/skill level
    skill_hier_info_col (str, optional): the taxonomy column name of the ids for the skill levels the skill belongs to
    skill_type_col (str): the taxonomy column name for which type of data the row is from (specific skill or skill levels)
    bert_model (str): sentence transformer
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
    map_skills(taxonomy, taxonomy_skills, ojo_skills_file_name, num_hier_levels, skill_type_dict):
            loads taxonomy and OJO skills; preprocesses skills; embeds
            and maps OJO onto taxonomy skills based on cosine similarity.
    """

    def __init__(
        self,
        taxonomy="esco",
        skill_name_col="description",
        skill_id_col="id",
        skill_hier_info_col=None,
        skill_type_col="type",
        bert_model=BertVectorizer().fit(),  # instantiate bert bembedding here
        job_ner=JobNER(),
        job_ner_model_folder="outputs/models/ner_model/20220729/",
    ):
        self.taxonomy = taxonomy
        self.skill_name_col = skill_name_col
        self.skill_id_col = skill_id_col
        self.skill_hier_info_col = skill_hier_info_col
        self.skill_type_col = skill_type_col
        self.bert_model = bert_model
        self.job_ner = job_ner
        self.job_ner_model_folder = job_ner_model_folder

    def load_job_skills(self, ojo_skills_file_name, s3=True):
        # load job skills here
        if s3:
            self.ojo_skills = load_s3_data(S3, bucket_name, ojo_skills_file_name)

        else:
            self.ojo_skills = load_json_dict(PROJECT_DIR / ojo_skills_file_name)

        return self.ojo_skills

    def preprocess_job_skills(self, ojo_skills=None):
        """
        ojo_skills: {'predictions': {'job_id': {'SKILL': , 'MULTISKILL': , 'EXPERIENCE': }, }
        """
        if not ojo_skills:
            ojo_skills = self.ojo_skills
        # preprocess skills
        self.ojo_job_ids = list(ojo_skills["predictions"].keys())
        self.clean_ojo_skills = dict()
        self.skill_hashes = dict()

        for ojo_job_id in self.ojo_job_ids:
            ojo_job_skills = ojo_skills["predictions"][ojo_job_id]["SKILL"]
            # ojo_job_multiskills = ojo_skills["predictions"][ojo_job_id]["MULTISKILL"] # TO DO
            if ojo_job_skills != []:
                self.clean_ojo_skills[ojo_job_id] = {
                    "clean_skills": list(
                        set([clean_text(skill) for skill in ojo_job_skills])
                    )
                }
            # create hashes of clean skills
            job_ad_skill_hashes = []
            if ojo_job_id in self.clean_ojo_skills.keys():
                for clean_skill in self.clean_ojo_skills[ojo_job_id]["clean_skills"]:
                    skill_hash = short_hash(clean_skill)
                    self.skill_hashes[skill_hash] = clean_skill
                    job_ad_skill_hashes.append(skill_hash)
                self.clean_ojo_skills[ojo_job_id]["skill_hashes"] = job_ad_skill_hashes

        return self.clean_ojo_skills, self.skill_hashes

    def load_taxonomy_skills(self, tax_input_file_name, s3=False):
        # load taxonomy skills
        if s3:
            self.taxonomy_skills = load_s3_data(S3, bucket_name, tax_input_file_name)
        else:
            self.taxonomy_skills = load_data(PROJECT_DIR / tax_input_file_name)

        # Sometimes the hierarchy list is read in as a string rather than a list,
        # so edit this if this happens
        def clean_string_list(string_list):
            if pd.notnull(string_list):
                if isinstance(string_list, str):
                    return ast.literal_eval(string_list)
                else:
                    return string_list
            else:
                return None

        if self.skill_hier_info_col:
            self.taxonomy_skills[self.skill_hier_info_col] = self.taxonomy_skills[
                self.skill_hier_info_col
            ].apply(clean_string_list)

        return self.taxonomy_skills

    def preprocess_taxonomy_skills(self, taxonomy_skills):
        # preprocess taxonomy skills
        taxonomy_skills["cleaned skills"] = taxonomy_skills[self.skill_name_col].apply(
            clean_text
        )[:10]

        taxonomy_skills.replace({np.nan: None}).reset_index(inplace=True, drop=True)

        return taxonomy_skills

    def embed_taxonomy_skills(self, taxonomy_skills):
        """embed and save clean taxonomy skills"""

        self.taxonomy_skills_embeddings = self.bert_model.transform(
            taxonomy_skills["cleaned skills"].to_list()
        )

        self.taxonomy_skills_embeddings_dict = dict(
            zip(taxonomy_skills.index, self.taxonomy_skills_embeddings)
        )

    def save_taxonomy_embeddings(self, taxonomy_embedding_file_name):
        save_to_s3(
            S3,
            bucket_name,
            self.taxonomy_skills_embeddings_dict,
            taxonomy_embedding_file_name,
        )

    def load_taxonomy_embeddings(self, taxonomy_embedding_file_name, s3=True):
        """Load taxonomy embeddings from s3"""
        if s3:
            self.taxonomy_skills_embeddings_dict = load_s3_data(
                S3, bucket_name, taxonomy_embedding_file_name
            )
        else:
            self.taxonomy_skills_embeddings_dict = load_json_dict(
                PROJECT_DIR / taxonomy_embedding_file_name
            )

        return self.taxonomy_skills_embeddings_dict

    def load_ojo_esco_mapper(self, ojo_esco_mapper_file_name, s3=True):
        """Load ojo esco mapper from s3"""
        if s3:
            self.ojo_esco = load_s3_data(S3, bucket_name, ojo_esco_mapper_file_name)
        else:
            self.ojo_esco = load_json_dict(PROJECT_DIR / ojo_esco_mapper_file_name)

        return self.ojo_esco

    def save_ojo_esco_mapper(self, ojo_esco_mapper_file_name, skill_hash_to_esco):
        """Saves final predictions as ojo_esco mapper"""
        save_to_s3(S3, bucket_name, skill_hash_to_esco, ojo_esco_mapper_file_name)

    def filter_skill_hash(self, skill_hashes, ojo_esco):
        """Filters skill hashes for skills not in ojo esco look up table."""
        self.skill_hashes_filtered = {
            skill_hash: skill
            for skill_hash, skill in skill_hashes.items()
            if skill_hash not in ojo_esco.keys()
        }

        return self.skill_hashes_filtered

    def map_skills(
        self, taxonomy_skills, skill_hashes_filtered, num_hier_levels, skill_type_dict
    ):
        """
        Maps skills not in ojo to esco look up dictionary


        taxonomy_skills (pandas DataFrame)
        num_hier_levels (int): the number of levels there are in this taxonomy
        skill_type_dict (dict):
                A dictionary of the values of the skill_type_col column which fit into either the skill_types or the hier_types
                e.g. {'skill_types': ['preferredLabel', 'altLabels'], 'hier_types': ["level_2", "level_3"],}
        """
        clean_ojo_skill_embeddings = self.bert_model.transform(
            skill_hashes_filtered.values()
        )
        # Find the closest matches to skills information
        skill_types = skill_type_dict.get("skill_types", [])
        tax_skills_ix = taxonomy_skills[
            taxonomy_skills[self.skill_type_col].isin(skill_types)
        ].index
        (skill_top_sim_indxs, skill_top_sim_scores) = get_top_comparisons(
            clean_ojo_skill_embeddings,
            [self.taxonomy_skills_embeddings_dict[i] for i in tax_skills_ix],
            match_sim_thresh=0.5,
        )
        # Find the closest matches to the hierarchy levels information
        hier_types = {i: v for i, v in enumerate(skill_type_dict.get("hier_types", []))}
        hier_types_top_sims = {}
        for hier_type_num, hier_type in hier_types.items():
            taxonomy_skills_ix = taxonomy_skills[
                taxonomy_skills[self.skill_type_col] == hier_type
            ].index
            top_sim_indxs, top_sim_scores = get_top_comparisons(
                clean_ojo_skill_embeddings,
                [self.taxonomy_skills_embeddings_dict[i] for i in taxonomy_skills_ix],
            )
            hier_types_top_sims[hier_type_num] = {
                "top_sim_indxs": top_sim_indxs,
                "top_sim_scores": top_sim_scores,
                "taxonomy_skills_ix": taxonomy_skills_ix,
            }
        # Output the top matches (using the different metrics) for each OJO skill
        # Need to match indexes back correctly (hence all the ix variables)
        self.skill_mapper_list = []
        for i, (match_i, match_text) in enumerate(skill_hashes_filtered.items()):
            # Top highest matches (any threshold)
            match_results = {
                "ojo_skill_id": match_i,
                "ojo_ner_skill": match_text,
                "top_tax_skills": list(
                    zip(
                        [
                            taxonomy_skills.iloc[tax_skills_ix[top_ix]][
                                self.skill_name_col
                            ]
                            for top_ix in skill_top_sim_indxs[i]
                        ],
                        [
                            taxonomy_skills.iloc[tax_skills_ix[top_ix]][
                                self.skill_id_col
                            ]
                            for top_ix in skill_top_sim_indxs[i]
                        ],
                        skill_top_sim_scores[i],
                    )
                ),
            }
            # Using the top matches, find the most common codes for each level of the
            # hierarchy (if hierarchy details are given), weighted by their similarity score
            if self.skill_hier_info_col:
                high_hier_codes = []
                for sim_ix, sim_score in zip(
                    skill_top_sim_indxs[i], skill_top_sim_scores[i]
                ):
                    tax_info = taxonomy_skills.iloc[tax_skills_ix[sim_ix]]
                    if tax_info[self.skill_hier_info_col]:
                        hier_levels = tax_info[self.skill_hier_info_col]
                        for hier_level in hier_levels:
                            high_hier_codes += [hier_level] * round(sim_score * 10)
                high_tax_skills_results = {}
                for hier_level in range(num_hier_levels):
                    high_tax_skills_results[
                        "most_common_level_" + str(hier_level)
                    ] = get_most_common_code(high_hier_codes, hier_level)

                match_results["high_tax_skills"] = high_tax_skills_results
            # Now get the top matches using the hierarchy descriptions (if hier_types isnt empty)
            for hier_type_num, hier_type in hier_types.items():
                hier_sims_info = hier_types_top_sims[hier_type_num]
                taxonomy_skills_ix = hier_sims_info["taxonomy_skills_ix"]
                tax_info = taxonomy_skills.iloc[
                    taxonomy_skills_ix[hier_sims_info["top_sim_indxs"][i][0]]
                ]
                match_results["top_" + hier_type + "_tax_level"] = (
                    tax_info[self.skill_name_col],
                    tax_info[self.skill_id_col],
                    hier_sims_info["top_sim_scores"][i][0],
                )

            self.skill_mapper_list.append(match_results)

        return self.skill_mapper_list

    def final_prediction(
        self,
        skills_to_taxonomy,
        hier_name_mapper,
        match_thresholds_dict,
        num_hier_levels,
    ):
        """

        Using all the information in skill_mapper_list get a final ESCO match (if any)
        for each ojo skill, based off a set of rules.
        """

        self.rank_matches = []
        for match_id, v in enumerate(skills_to_taxonomy):
            match_num = 0

            # Try to find a close similarity skill
            skill_info = {
                "ojo_skill": v["ojo_ner_skill"],
                "match_id": v["ojo_skill_id"],
            }
            match_hier_info = {}
            top_skill, top_skill_code, top_sim_score = v["top_tax_skills"][0]
            if top_sim_score >= match_thresholds_dict["skill_match_thresh"]:
                skill_info.update({"match " + str(match_num): top_skill})
                match_hier_info[match_num] = {
                    "match_code": top_skill_code,
                    "type": "skill",
                    "value": top_sim_score,
                }
                match_num += 1

            # Go through hierarchy levels from most granular to least
            # and try to find a close match first in the most common level then in
            # the level name with the closest similarity
            for n in reversed(range(num_hier_levels)):
                # Look at level n most common
                type_name = "most_common_level_" + str(n)
                if (type_name in v["high_tax_skills"]) and (
                    n in match_thresholds_dict["max_share"]
                ):
                    c0 = v["high_tax_skills"][type_name]
                    if (c0[1]) and (c0[1] >= match_thresholds_dict["max_share"][n]):
                        match_name = hier_name_mapper.get(c0[0], c0[0])
                        skill_info.update({"match " + str(match_num): match_name})
                        match_hier_info[match_num] = {
                            "match_code": c0[0],
                            "type": type_name,
                            "value": c0[1],
                        }
                        match_num += 1

                # Look at level n closest similarity
                type_name = "top_level_" + str(n) + "_tax_level"
                if (type_name in v) and (n in match_thresholds_dict["top_tax_skills"]):
                    c1 = v[type_name]
                    if c1[2] >= match_thresholds_dict["top_tax_skills"][n]:
                        skill_info.update({"match " + str(match_num): c1[0]})
                        match_hier_info[match_num] = {
                            "match_code": c1[1],
                            "type": type_name,
                            "value": c1[2],
                        }
                        match_num += 1

            skill_info.update({"match_info": match_hier_info})
            self.rank_matches.append(skill_info)

        # Just pull out the top matches for each ojo skill
        self.final_match = []
        for rank_match in self.rank_matches:
            self.final_match.append(
                {
                    "ojo_skill": rank_match["ojo_skill"],
                    "ojo_job_skill_hash": rank_match["match_id"],
                    "match_skill": rank_match["match 0"],
                    "match_score": rank_match["match_info"][0]["value"],
                    "match_type": rank_match["match_info"][0]["type"],
                    "match_id": rank_match["match_info"][0]["match_code"],
                }
            )

        return self.final_match

    def append_final_predictions(self, final_match, ojo_esco):
        """Append ojo to esco look up to the final predictions."""
        for skill_hash, esco_info in ojo_esco.items():
            esco_info["ojo_job_skill_hash"] = skill_hash

        return list(ojo_esco.values()) + self.final_match

    def link_skill_hash_to_job_id(self, clean_ojo_skills, final_matches):
        """Append ojo to esco look up to the final predictions."""
        self.skill_hash_to_esco = {}
        for fm in final_matches:
            self.skill_hash_to_esco[fm["ojo_job_skill_hash"]] = fm

        for job_id, job_info in self.clean_ojo_skills.items():
            esco_skills = []
            for skill_hash in job_info["skill_hashes"]:
                esco_skills.append(self.skill_hash_to_esco[skill_hash])
            job_info["skill_to_taxonomy"] = esco_skills

        return self.skill_hash_to_esco, self.clean_ojo_skills


if __name__ == "__main__":

    set_global_logging_level(level=logging.ERROR, prefices=["sentence_transformers", "boto"])
    parser = ArgumentParser()

    parser.add_argument(
        "--ojo_skill_fn",
        help="Name of ojo skills file name to be mapped to.",
        default=config["ojo_skills_ner_path"],
    )

    args = parser.parse_args()
    ojo_skill_file_name = args.ojo_skill_fn

    skill_mapper = SkillMapper(
        skill_name_col="description",
        skill_id_col="id",
        skill_hier_info_col="hierarchy_levels",
        skill_type_col="type",
    )
    # Hard code how many levels there are in the taxonomy (if any)
    # This should correspond to the length of the data in taxonomy_skills["hierarchy_levels"] e.g. ['S', S4', 'S4.8, 'S4.8.1']
    if skill_mapper.taxonomy == "esco":  # put in a config file at some point
        num_hier_levels = 4
        skill_type_dict = {
            "skill_types": ["preferredLabel", "altLabels"],
            "hier_types": ["level_2", "level_3"],
        }
        tax_input_file_name = (
            "escoe_extension/outputs/data/skill_ner_mapping/esco_data_formatted.csv"
        )
        match_thresholds_dict = {
            "skill_match_thresh": 0.7,
            "top_tax_skills": {1: 0.5, 2: 0.5, 3: 0.5},
            "max_share": {1: 0, 2: 0.2, 3: 0.2},
        }
        hier_name_mapper_file_name = (
            "escoe_extension/outputs/data/skill_ner_mapping/esco_hier_mapper.json"
        )
        esco_embeddings_file_name = (
            "escoe_extension/outputs/data/skill_ner_mapping/esco_embeddings.json"
        )
        ojo_esco_lookup_file_name = (
            "escoe_extension/outputs/data/skill_ner_mapping/ojo_esco_lookup.json"
        )

    else:
        num_hier_levels = 0
        skill_type_dict = {}
        tax_input_file_name = ""
        esco_embeddings_file_name = ""
        ojo_esco_lookup_file_name = ""

    ojo_skills = skill_mapper.load_job_skills(config["ojo_skills_ner_path"], s3=True)
    logger.info("loaded ojo skills")
    clean_ojo_skills, skill_hashes = skill_mapper.preprocess_job_skills(ojo_skills)
    logger.info("cleaned ojo skills and skill hashes")

    taxonomy_skills = skill_mapper.load_taxonomy_skills(tax_input_file_name, s3=True)
    logger.info("loaded taxonomy skills")
    taxonomy_skills = skill_mapper.preprocess_taxonomy_skills(taxonomy_skills)
    logger.info("preprocessed taxonomy skills")

    embedding_lookup_files = get_s3_data_paths(
        S3,
        bucket_name,
        "escoe_extension/outputs/data/skill_ner_mapping/",
        file_types=["*.json"],
    )

    if esco_embeddings_file_name in embedding_lookup_files:
        taxonomy_embeddings = skill_mapper.load_taxonomy_embeddings(
            esco_embeddings_file_name
        )
    else:
        skill_mapper.embed_taxonomy_skills(
            taxonomy_skills,
        )
        skill_mapper.save_taxonomy_embeddings(esco_embeddings_file_name)
        taxonomy_embeddings = skill_mapper.load_taxonomy_embeddings(
            esco_embeddings_file_name
        )
    logger.info("loaded esco embeddings")

    if ojo_esco_lookup_file_name in embedding_lookup_files:
        ojo_esco_predefined = skill_mapper.load_ojo_esco_mapper(
            ojo_esco_lookup_file_name
        )
        skill_hashes = skill_mapper.filter_skill_hash(skill_hashes, ojo_esco_predefined)
        logger.info("loaded skill hashes file")
    else:
        ojo_esco_predefined = None

    logger.info("no ojo esco look up file - just use all skill hashes")

    skills_to_taxonomy = skill_mapper.map_skills(
        taxonomy_skills,
        skill_hashes,
        num_hier_levels=num_hier_levels,
        skill_type_dict=skill_type_dict,
    )

    full_skill_mapper_file_name = (
        ojo_skill_file_name.split("/")[-1].split(".")[0]
        + "_to_"
        + skill_mapper.taxonomy
        + "_full_matches.json"
    )

    save_to_s3(
        get_s3_resource(),
        bucket_name,
        skills_to_taxonomy,
        os.path.join(config["ojo_skills_ner_mapping_dir"], full_skill_mapper_file_name),
    )

    # Get the final result - one match per OJO skill
    hier_name_mapper = load_s3_data(S3, bucket_name, hier_name_mapper_file_name)

    final_matches = skill_mapper.final_prediction(
        skills_to_taxonomy, hier_name_mapper, match_thresholds_dict, num_hier_levels
    )

    # append final lookup to predictions
    if ojo_esco_predefined:
        final_matches = skill_mapper.append_final_predictions(
            final_matches, ojo_esco_predefined
        )
        (
            skill_hash_to_esco,
            final_ojo_skills,
        ) = skill_mapper.link_skill_hash_to_job_id(clean_ojo_skills, final_matches)
    else:
        (
            skill_hash_to_esco,
            final_ojo_skills,
        ) = skill_mapper.link_skill_hash_to_job_id(clean_ojo_skills, final_matches)
        # and save final matches as mapper
        skill_mapper.save_ojo_esco_mapper(skill_hash_to_esco, ojo_esco_lookup_file_name)

#    skill_mapper_file_name = (
#        ojo_skill_file_name.split("/")[-1].split(".")[0] + "_to_" + taxonomy + ".json"
#    )

# save_to_s3(
#     S3,
#     bucket_name,
#     final_matches,
#     os.path.join(config["ojo_skills_ner_mapping_dir"], skill_mapper_file_name),
# )
