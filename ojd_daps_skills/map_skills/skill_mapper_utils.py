"""
Utility functions to map extracted skills from NER model to
taxonomy skills and configuration managers for mapping skills. 
"""

import ast
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import srsly
import yaml
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from wasabi import msg

from ojd_daps_skills import PROJECT_DIR, PUBLIC_DATA_FOLDER_PATH
from ojd_daps_skills.utils.bert_vectorizer import BertVectorizer
from ojd_daps_skills.utils.download_public_data import download_data


def get_top_comparisons(ojo_embs: np.array, taxonomy_embs: np.array) -> Tuple[list]:
    """Get the top 10 most similar taxonomy skills for each extracted skill.

    Args:
        ojo_embs (np.array): Embeddings of extracted skills.
        taxonomy_embs (np.array): Embeddings of taxonomy skills.

    Returns:
        Tuple[list]: List of top 10 most similar taxonomy skills
        for each extracted skill and their corresponding scores.
    """
    if ojo_embs.size > 0:
        emb_sims = cosine_similarity(ojo_embs, taxonomy_embs)

        top_sim_indxs = [list(np.argsort(sim)[::-1][:10]) for sim in emb_sims]
        top_sim_scores = [
            [float(s) for s in np.sort(sim)[::-1][:10]] for sim in emb_sims
        ]

        return top_sim_indxs, top_sim_scores

    else:

        return None, None


def get_most_common_code(
    split_possible_codes: List[List[str]], lev_n: int
) -> Union[Tuple[str, float], Tuple[None, None]]:
    """Calculate the proportion of skills at a given level in the taxonomy.

    e.g.
    split_possible_codes = [['S4', 'S4.8', 'S4.8.1'],['S1', 'S1.8', 'S1.8.1'],['S1', 'S1.8', 'S1.8.1'], ['S1', 'S1.12', 'S1.12.3']]
    lev_n = 0
    will output ('S1', 0.75) [i.e. 'S1' is 75% of the level 0 codes]

    Args:
        split_possible_codes (List[List[str]]): List of lists of possible codes.
        lev_n (int): Level of the taxonomy to calculate the proportion for.

    Returns:
        Union[Tuple[str, float], Tuple[None, None]]: Tuple of the most common code at the given level
            and the proportion of skills at that level. If no codes are found, returns None, None.
    """

    if any([isinstance(el, str) for el in split_possible_codes]):
        split_possible_codes = [split_possible_codes]

    lev_codes = [w[lev_n] for w in split_possible_codes if w[lev_n]]
    if lev_codes:
        lev_code, lev_num = Counter(lev_codes).most_common(1)[0]
        lev_prop = (
            0 if len(split_possible_codes) == 0 else lev_num / len(split_possible_codes)
        )
        return lev_code, lev_prop
    else:
        return None, None


def get_top_match(
    score_0: Union[None, float],
    score_1: Union[None, float],
    threshold_0: float,
    threshold_1: float,
):
    """Get the top match between two scores based on thresholds.

    Args:
        score_0 (Union[None, float]): A score.
        score_1 (Union[None, float]): A score.
        threshold_0 (float): A threshold.
        threshold_1 (float): A threshold.

    Returns:
        The top match between two scores based on thresholds.
    """

    if not score_0:
        score_0 = 0
    if not score_1:
        score_1 = 0

    if score_0 < threshold_0:
        if score_1 < threshold_1:
            return None
        else:
            return 1
    else:
        if score_1 < threshold_1:
            return 0
        else:
            return np.argmax([score_0, score_1])


def _clean_string_list(string_list: str) -> Union[List[str], None]:
    """Convert string list to list.

    Args:
        string_list (str): String list.

    Returns:
        Union[List[str], None]: List of strings or None.
    """
    if pd.notnull(string_list):
        if isinstance(string_list, str):
            return ast.literal_eval(string_list)
        else:
            return string_list
    else:
        return None


class MapConfig(BaseModel):
    """
    Configuration manager for MAPPING skills to pre-defined taxonomies.

    Attributes:
        taxonomy_name (str): The name of the taxonomy to use. Current configuration supports
            "esco", "lightcast" or "toy" for testing purposes.
        taxonomy_config (Dict[str, Any]): Config associated to the taxonomy. This includes
            information like the column names of the taxonomy data, thresholding values
            at different levels of the taxonomy, etc.
        bert_model (BertVectorizer): The BERT model used for vectorizing skills to
            calculate similarity scores between extracted skills and taxonomy skills.
        taxonomy_data (pd.DataFrame): The taxonomy data to use for mapping skills. This
            includes the skill names, skill descriptions, and hierarchical information.
        taxonomy_embeddings (Optional[Dict[int, np.array]]): The embeddings of the taxonomy
            data. This is used to calculate similarity scores between extracted skills and
            taxonomy skills.
        hier_mapper (Dict[str, str]): A dictionary mapping the hierarchical information of
            the taxonomy data.
        hard_coded_taxonomy (Optional[Dict[int, dict]]): A hard-coded taxonomy lookup for
            specific taxonomies. This is used for taxonomies like ESCO where we have already
            identified the most appropriate skill matches.
        previous_skill_matches (Optional[Dict[int, str]]): A dictionary of previous skill
            matches for specific taxonomies. This is used for taxonomies like ESCO where we
            have already identified the most appropriate skill matches.
        match_sim_thresh (float): The similarity threshold to use when matching extracted
            skills to taxonomy skills.
    """

    taxonomy_name: Optional[str] = None
    taxonomy_config: Optional[Dict[str, Any]] = None
    bert_model: Optional[BertVectorizer] = None
    taxonomy_data: Optional[pd.DataFrame] = None
    taxonomy_embeddings: Optional[Dict[int, np.array]] = None
    hier_mapper: Optional[Dict[str, str]] = None
    hard_coded_taxonomy: Optional[Dict[int, Any]] = None
    previous_skill_matches: Optional[Dict[int, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls, taxonomy_name: Optional[str] = "toy") -> "MapConfig":
        """
        Creates an instance of MapConfig by loading configurations.

        Parameters:
           taxonomy_name (str): The name of the taxonomy to use. Current configuration supports
            "esco", "lightcast" or "toy" for testing purposes.

        Returns:
            MapConfig: An initialized instance of this configuration class.

        Raises:
            msg.fail: If the configuration file or data is not loaded locally, this error
                is raised.
        """
        config_path = PROJECT_DIR / "ojd_daps_skills/configs"
        config_file = config_path / f"extract_skills_{taxonomy_name}.yaml"

        # Load configuration file
        if not config_file.exists():
            raise msg.fail(f"Configuration file not found: {config_file}", exits=1)

        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file)

        # Load data
        if not PUBLIC_DATA_FOLDER_PATH.exists():
            msg.fail(
                f"Neccessary data files are not downloaded. Downloading ~0.5GB of neccessary data files to {PUBLIC_DATA_FOLDER_PATH}."
            )
            download_data()
        else:
            msg.good(f"Data files are already downloaded to {PUBLIC_DATA_FOLDER_PATH}.")

        multi_process = False
        bert_model = BertVectorizer(multi_process=multi_process).fit()

        # taxonomy information
        taxonomy_data_path = (
            PUBLIC_DATA_FOLDER_PATH / f"{taxonomy_name}_data_formatted.csv"
        )
        if taxonomy_data_path.exists():
            taxonomy_data = pd.read_csv(
                PUBLIC_DATA_FOLDER_PATH / f"{taxonomy_name}_data_formatted.csv"
            )
            taxonomy_data = taxonomy_data[
                taxonomy_data[config_data["skill_name_col"]].notna()
            ].reset_index(drop=True)

            if config_data["skill_hier_info_col"]:
                taxonomy_data[config_data["skill_hier_info_col"]] = taxonomy_data[
                    config_data["skill_hier_info_col"]
                ].apply(_clean_string_list)

        else:
            raise msg.fail(f"Taxonomy data not found: {taxonomy_data_path}", exits=1)

        taxonomy_embeddings_path = (
            PUBLIC_DATA_FOLDER_PATH / f"{taxonomy_name}_embeddings.json"
        )
        if taxonomy_embeddings_path.exists():
            taxonomy_embeddings = srsly.read_json(
                PUBLIC_DATA_FOLDER_PATH / f"{taxonomy_name}_embeddings.json"
            )
            taxonomy_embeddings = {
                int(k): np.array(v) for k, v in taxonomy_embeddings.items()
            }
        else:
            taxonomy_embeddings = None

        hier_mapper_path = PUBLIC_DATA_FOLDER_PATH / f"{taxonomy_name}_hier_mapper.json"
        if hier_mapper_path.exists():
            hier_mapper = srsly.read_json(
                PUBLIC_DATA_FOLDER_PATH / f"{taxonomy_name}_hier_mapper.json"
            )
        else:
            msg.fail(f"Hierarchical mapper not found: {hier_mapper_path}", exits=1)
        # here, let's download the hard-coded taxonomy if it's for escoe
        if taxonomy_name == "esco":
            hard_coded_taxonomy = srsly.read_json(
                PUBLIC_DATA_FOLDER_PATH / f"hardcoded_ojo_{taxonomy_name}_lookup.json"
            )
            previous_skill_matches = srsly.read_json(
                PUBLIC_DATA_FOLDER_PATH / f"ojo_{taxonomy_name}_lookup_sample.json"
            )

        else:
            hard_coded_taxonomy = None  # no hard coded taxonomy for other taxonomies
            previous_skill_matches = (
                None  # no previous skill matches for other taxonomies
            )

        return cls(
            taxonomy_name=taxonomy_name,
            taxonomy_config=config_data,
            bert_model=bert_model,
            taxonomy_data=taxonomy_data,
            taxonomy_embeddings=taxonomy_embeddings,
            hier_mapper=hier_mapper,
            hard_coded_taxonomy=hard_coded_taxonomy,
            previous_skill_matches=previous_skill_matches,
        )
