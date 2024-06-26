"""
SkillsMapper class to MAP extracted skills from job ads. 
"""

from itertools import chain
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel
from spacy.tokens import Doc

from ojd_daps_skills import setup_spacy_extensions
from ojd_daps_skills.map_skills.skill_mapper_utils import (
    MapConfig,
    get_most_common_code,
    get_top_comparisons,
)
from ojd_daps_skills.utils.text_cleaning import clean_text, short_hash

setup_spacy_extensions()


class SkillsMapper(BaseModel):
    """
    SkillsMapper class to MAP extracted skills from job ads.

    It takes a rules-based semantic similarity approach
    to map skills to a pre-defined skills taxonomy
    based on the hierarchy of a taxonomy and the
    similarity of the skill embeddings.

    Attributes:
        taxonomy_name (str): The name of the taxonomy. Default is "toy".
        config (MapConfig): A configuration manager for mapping skills.
            It is initiated with the taxonomy name.
        all_skills_unique_dict (Dict[int, str]): A dictionary with unique skill
            hashes as keys and the corresponding skill text as values. It is
            created during the get_embeddings method.
    """

    config: MapConfig
    all_skills_unique_dict: Dict[int, str] = {}

    def get_top_taxonomy_skills(
        self,
        skill_embeddings: np.ndarray,
        taxonomy_embeddings_dict: Dict[int, np.array],
    ) -> Tuple[List[int], List[float]]:
        """Get the top taxonomy skills at the lowest level of the taxonomy.
        Args:
            skill_embeddings (np.ndarray): An array of skill semantic embeddings.
            taxonomy_embeddings_dict (Dict[int, np.array]): A dictionary with
                integer keys and semantic embeddings as values.

        Returns:
            Tuple[List[int], List[float]]: A tuple of the top taxonomy skill indices
                and the corresponding similarity scores.
        """
        skill_types = self.config.taxonomy_config["skill_type_dict"].get(
            "skill_types", []
        )
        tax_skills_ix = self.config.taxonomy_data[
            self.config.taxonomy_data[
                self.config.taxonomy_config["skill_type_col"]
            ].isin(skill_types)
        ].index

        # here, we map at the lowest level of the taxonomy first
        (skill_top_sim_indxs, skill_top_sim_scores) = get_top_comparisons(
            skill_embeddings,
            [taxonomy_embeddings_dict[i] for i in tax_skills_ix],
        )

        return skill_top_sim_indxs, skill_top_sim_scores, tax_skills_ix

    def get_top_hierarchy_skills(
        self,
        skill_embeddings: np.ndarray,
        taxonomy_embeddings_dict: Dict[int, np.array],
    ) -> Dict[int, Dict[str, List[int]]]:
        """Get the top taxonomy skills at each level of the taxonomy.

        Args:
            skill_embeddings (np.ndarray): An array of skill semantic embeddings.
            taxonomy_embeddings_dict (Dict[int, np.array]): A dictionary with
                integer keys and semantic embeddings as values.

        Returns:
            Dict[int, Dict[str, List[int]]]: A dictionary with integer keys
                and a dictionary with semantically similar skill indices
                at each level of the taxonomy.
        """
        hier_types = {
            i: v
            for i, v in enumerate(
                self.config.taxonomy_config["skill_type_dict"].get("hier_types", [])
            )
        }

        hier_types_top_sims = {}
        for hier_type_num, hier_type in hier_types.items():
            taxonomy_skills_ix = self.config.taxonomy_data[
                self.config.taxonomy_data[self.config.taxonomy_config["skill_type_col"]]
                == hier_type
            ].index
            top_sim_indxs, top_sim_scores = get_top_comparisons(
                skill_embeddings,
                [taxonomy_embeddings_dict[i] for i in taxonomy_skills_ix],
            )
            hier_types_top_sims[hier_type_num] = {
                "top_sim_indxs": top_sim_indxs,
                "top_sim_scores": top_sim_scores,
                "taxonomy_skills_ix": taxonomy_skills_ix.to_list(),
            }

        return hier_types_top_sims, hier_types

    def get_embeddings(
        self, job_ads: List[Doc]
    ) -> Tuple[np.ndarray, Dict[int, np.array]]:
        """Get the embeddings for all unique skills in the job ads and the taxonomy.

        Args:
            job_ads (List[Doc]): A list of spaCy Doc objects with skill spans.

        Returns:
            Tuple[np.ndarray, Dict[int, np.array]]: A tuple of the skill embeddings
                and a dictionary with taxonomy skill indices and embeddings.
        """
        all_skills = list(chain.from_iterable([doc._.skill_spans for doc in job_ads]))
        all_skills_unique = list(set(all_skills))

        if not isinstance(self.config.hard_coded_taxonomy, dict):
            self.config.hard_coded_taxonomy = {}

        self.all_skills_unique_dict = {}
        for skill in all_skills_unique:
            skill_clean = clean_text(skill)
            skill_hash = short_hash(skill_clean)
            if not self.config.hard_coded_taxonomy.get(skill_hash):
                self.all_skills_unique_dict[skill_hash] = skill_clean

        skill_embeddings = self.config.bert_model.transform(
            list(self.all_skills_unique_dict.values())
        )

        if not self.config.taxonomy_embeddings:
            taxonomy_embeddings = self.config.bert_model.transform(
                self.config.taxonomy_data[self.config.skill_name_col].to_list()
            )
            taxonomy_embeddings_dict = dict(
                zip(
                    self.config.taxonomy_data.index.astype("int").to_list(),
                    taxonomy_embeddings,
                )
            )

        else:
            taxonomy_embeddings_dict = self.config.taxonomy_embeddings

        return skill_embeddings, taxonomy_embeddings_dict

    def map_skills(self, job_ads: List[Doc]) -> List[Dict[str, Any]]:
        """Map the skills extracted from the job ads to the taxonomy.

        Args:
            job_ads (List[Doc]): A list of spaCy Doc objects with skill spans.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with the mapped skills.
        """

        skill_embeddings, taxonomy_embeddings_dict = self.get_embeddings(job_ads)

        (
            top_skill_indxs,
            top_skill_scores,
            tax_skills_ix,
        ) = self.get_top_taxonomy_skills(skill_embeddings, taxonomy_embeddings_dict)

        if self.config.taxonomy_config.get("skill_hier_info_col"):
            top_hier_skills, hier_types = self.get_top_hierarchy_skills(
                skill_embeddings, taxonomy_embeddings_dict
            )

        # Output the top matches (using the different metrics) for each OJO skill
        # Need to match indexes back correctly (hence all the ix variables)
        skill_mapper_list = []

        for i, (match_i, match_text) in enumerate(self.all_skills_unique_dict.items()):
            # Top highest matches (any threshold)
            match_results = {
                "ojo_skill_id": match_i,
                "ojo_ner_skill": match_text,
                "top_tax_skills": list(
                    zip(
                        [
                            self.config.taxonomy_data.iloc[tax_skills_ix[top_ix]][
                                self.config.taxonomy_config["skill_name_col"]
                            ]
                            for top_ix in top_skill_indxs[i]
                        ],
                        [
                            self.config.taxonomy_data.iloc[tax_skills_ix[top_ix]][
                                self.config.taxonomy_config["skill_id_col"]
                            ]
                            for top_ix in top_skill_indxs[i]
                        ],
                        top_skill_scores[i],
                    )
                ),
            }
            # Using the top matches, find the most common codes for each level of the
            # hierarchy (if hierarchy details are given), weighted by their similarity score
            if self.config.taxonomy_config.get("skill_hier_info_col"):
                high_hier_codes = []
                for sim_ix, sim_score in zip(top_skill_indxs[i], top_skill_scores[i]):
                    tax_info = self.config.taxonomy_data.iloc[tax_skills_ix[sim_ix]]
                    if tax_info[self.config.taxonomy_config["skill_hier_info_col"]]:
                        hier_levels = tax_info[
                            self.config.taxonomy_config["skill_hier_info_col"]
                        ]
                        for hier_level in hier_levels:
                            high_hier_codes += [hier_level] * round(sim_score * 10)
                high_tax_skills_results = {}
                for hier_level in range(self.config.taxonomy_config["num_hier_levels"]):
                    high_tax_skills_results["most_common_level_" + str(hier_level)] = (
                        get_most_common_code(high_hier_codes, hier_level)
                    )

                if high_tax_skills_results:
                    match_results["high_tax_skills"] = high_tax_skills_results
            # Now get the top matches using the hierarchy descriptions (if hier_types isnt empty)
            for hier_type_num, hier_type in hier_types.items():
                hier_sims_info = top_hier_skills[hier_type_num]
                taxonomy_skills_ix = hier_sims_info["taxonomy_skills_ix"]
                tax_info = self.config.taxonomy_data.iloc[
                    taxonomy_skills_ix[hier_sims_info["top_sim_indxs"][i][0]]
                ]
                match_results["top_" + hier_type + "_tax_level"] = (
                    tax_info[self.config.taxonomy_config["skill_name_col"]],
                    tax_info[self.config.taxonomy_config["skill_id_col"]],
                    hier_sims_info["top_sim_scores"][i][0],
                )

            skill_mapper_list.append(match_results)

        return skill_mapper_list

    def match_skills(self, job_ads: List[Doc]) -> Dict[int, dict]:
        """Rules-based matching of skills to the taxonomy.

        Args:
            job_ads (List[Doc]): A list of spaCy Doc objects with skill spans.

        Returns:
            Dict[int, dict]: A dictionary with the matched skills.
        """

        mapped_skills = self.map_skills(job_ads)

        rank_matches = []
        for _, v in enumerate(mapped_skills):
            match_num = 0

            # Try to find a close similarity skill
            skill_info = {
                "ojo_skill": v["ojo_ner_skill"],
                "match_id": v["ojo_skill_id"],
            }
            match_hier_info = {}
            top_skill, top_skill_code, top_sim_score = v["top_tax_skills"][0]
            if (
                top_sim_score
                >= self.config.taxonomy_config["match_thresholds_dict"][
                    "skill_match_thresh"
                ]
            ):
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
            for n in reversed(range(self.config.taxonomy_config["num_hier_levels"])):
                # Look at level n most common
                type_name = "most_common_level_" + str(n)
                if "high_tax_skills" in v.keys():
                    if (type_name in v["high_tax_skills"]) and (
                        n
                        in self.config.taxonomy_config["match_thresholds_dict"][
                            "max_share"
                        ]
                    ):
                        c0 = v["high_tax_skills"][type_name]
                        if (c0[1]) and (
                            c0[1]
                            >= self.config.taxonomy_config["match_thresholds_dict"][
                                "max_share"
                            ][n]
                        ):
                            match_name = self.config.hier_mapper.get(c0[0], c0[0])
                            skill_info.update({"match " + str(match_num): match_name})
                            match_hier_info[match_num] = {
                                "match_code": c0[0],
                                "type": type_name,
                                "value": c0[1],
                            }
                            match_num += 1

                # Look at level n closest similarity
                type_name = "top_level_" + str(n) + "_tax_level"
                if (type_name in v) and (
                    n
                    in self.config.taxonomy_config["match_thresholds_dict"][
                        "top_tax_skills"
                    ]
                ):
                    c1 = v[type_name]
                    if (
                        c1[2]
                        >= self.config.taxonomy_config["match_thresholds_dict"][
                            "top_tax_skills"
                        ][n]
                    ):
                        skill_info.update({"match " + str(match_num): c1[0]})
                        match_hier_info[match_num] = {
                            "match_code": c1[1],
                            "type": type_name,
                            "value": c1[2],
                        }
                        match_num += 1

            skill_info.update({"match_info": match_hier_info})
            rank_matches.append(skill_info)

        # Just pull out the top matches for each ojo skill
        final_match = []
        for rank_match in rank_matches:
            if "match 0" in rank_match.keys():
                final_match_dict = {
                    "ojo_skill": rank_match["ojo_skill"],
                    "ojo_skill_id": rank_match["match_id"],
                    "match_skill": rank_match["match 0"],
                    "match_score": rank_match["match_info"][0]["value"],
                    "match_type": rank_match["match_info"][0]["type"],
                    "match_id": rank_match["match_info"][0]["match_code"],
                }
                final_match.append(final_match_dict)

        final_match_dict = {match["ojo_skill_id"]: match for match in final_match}

        if self.config.hard_coded_taxonomy:
            final_match_dict = {**final_match_dict, **self.config.hard_coded_taxonomy}

        return final_match_dict
