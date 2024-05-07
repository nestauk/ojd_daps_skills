"""
Utility functions and configuration managers for 
extracting skills from job descriptions.
"""

import ast
import os
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import spacy
import srsly
import yaml

from ... import PROJECT_DIR, PUBLIC_DATA_FOLDER_NAME
from ..utils.download_public_data import download_data
from ..utils.bert_vectorizer import BertVectorizer

from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from skops.hub_utils import download
from spacy.tokens import Doc
from wasabi import msg

PUBLIC_DATA_FOLDER_PATH = PROJECT_DIR / PUBLIC_DATA_FOLDER_NAME


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


### Define Multiskill Transformer
class MultiSkillTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting necessary, just return self
        return self

    def transform(self, X):
        """Apply the transform_skill function to each element in X.

        Args:
            X (iterable of str): The data to transform.

        Returns:
            List[List[int]]: Transformed data, where each item is the output of transform_skill.
        """
        return [self.transform_skill(skill) for skill in X]

    @staticmethod
    def transform_skill(skill: str) -> List[int]:
        """Transform skill into a list of features. The features are:
            - length of skill span;
            - presence of " and " in skill span;
            - presence of "," in skill span.

        Args:
            skill (str): skill span.

        Returns:
            List[int]: list of integers.
        """
        return [len(skill), int(" and " in skill), int("," in skill)]


class ExtractConfig(BaseModel):
    """
    Configuration manager for EXTRACTING skills using specific NLP models.

    Attributes:
        ner_model_name (str): The name of the Named Entity Recognition model to use from HuggingFace Hub. Current
            configuration supports "nestauk/en_skillner".
            You can use your own NER model if you have a custom NER model to extract skills.
        ms_model_name (str): The name of the Multi-Skill model to use. Current configurations
            supports "nestauk/multiskill-classifier".
    """

    ner_model_name: str = "nestauk/en_skillner"
    ms_model_name: str = f"nestauk/multiskill-classifier"
    nlp: spacy.Language
    ms_model: Pipeline

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls, ner_model_name: str, ms_model_name: str):
        """
        Creates an instance of ExtractConfig by loading configurations.

        Parameters:
            ner_model_name (str): The name of the NER model to use.
            ms_model_name (str): The name of the Multi-Skill model to use.

        Returns:
            ExtractConfig: An initialized instance of this configuration class.

        Raises:
            msg.fail: If the data or Multi-Skill models are not loaded
            locally, this error is raised.
            OSError: If the NER model is not loaded, this error is raised.
        """
        # set Doc extension here
        if not Doc.has_extension("skill_spans"):
            Doc.set_extension("skill_spans", default=[])

        if "/" in ner_model_name:
            namespace, ner_name = ner_model_name.split("/")
        else:
            msg.fail(
                f"Invalid NER model name: {ner_model_name}. Must include HuggingFace namespace and model name.",
                exit=1,
            )
        try:
            nlp = spacy.load(ner_name)

        except OSError:
            msg.fail(f"{ner_model_name} NER model not loaded. Downloading model...")
            os.system(
                f"pip install https://huggingface.co/{namespace}/{ner_name}/resolve/main/{ner_name}-any-py3-none-any.whl"
            )

        # Load multi-skill model
        ms_model_path = PUBLIC_DATA_FOLDER_PATH / "models/ms_model"
        try:
            clf = joblib.load(ms_model_path / "multiskill-classifier8lnyq0he.pkl")
        except Exception:
            msg.fail("Multi-skill classifier not loaded. Downloading model...")
            download(repo_id=ms_model_name, dst=ms_model_path)
            clf = joblib.load(ms_model_path / "multiskill-classifier8lnyq0he.pkl")

        ms_model = Pipeline(
            [("transformer", MultiSkillTransformer()), ("classifier", clf)]
        )

        return cls(
            ner_model_name=ner_model_name,
            ms_model_name=ms_model_name,
            nlp=nlp,
            ms_model=ms_model,
        )


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

    taxonomy_name: str
    taxonomy_config: Dict[str, Any]
    bert_model: BertVectorizer
    taxonomy_data: pd.DataFrame
    taxonomy_embeddings: Optional[Dict[int, np.array]]
    hier_mapper: Dict[str, str]
    hard_coded_taxonomy: Optional[Dict[int, dict]]
    previous_skill_matches: Optional[Dict[int, str]]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls, taxonomy_name: str):
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

        config_path = PROJECT_DIR / "ojd_daps_skills/config"
        config_file = config_path / f"extract_skills_{taxonomy_name}.yaml"

        # Load configuration file
        if not config_file.exists():
            raise msg.fail(f"Configuration file not found: {config_file}", exits=1)

        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file)

        # Load data
        if not PUBLIC_DATA_FOLDER_PATH.exists():
            msg.fail(
                f"Neccessary data files are not downloaded. Downloading ~1GB of neccessary data files to {PUBLIC_DATA_FOLDER_PATH}."
            )
            download_data()
        else:
            msg.good(f"Data files are already downloaded to {PUBLIC_DATA_FOLDER_PATH}.")

        verbose = True
        multi_process = False
        bert_model = BertVectorizer(verbose=verbose, multi_process=multi_process).fit()

        # taxonomy information
        data_path = PUBLIC_DATA_FOLDER_PATH / "outputs/data/skill_ner_mapping"

        taxonomy_data_path = data_path / f"{taxonomy_name}_data_formatted.csv"
        if taxonomy_data_path.exists():
            taxonomy_data = pd.read_csv(
                data_path / f"{taxonomy_name}_data_formatted.csv"
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

        taxonomy_embeddings_path = data_path / f"{taxonomy_name}_embeddings.json"
        if taxonomy_embeddings_path.exists():
            taxonomy_embeddings = srsly.read_json(
                data_path / f"{taxonomy_name}_embeddings.json"
            )
            taxonomy_embeddings = {
                int(k): np.array(v) for k, v in taxonomy_embeddings.items()
            }
        else:
            taxonomy_embeddings = None

        hier_mapper_path = data_path / f"{taxonomy_name}_hier_mapper.json"
        if hier_mapper_path.exists():
            hier_mapper = srsly.read_json(
                data_path / f"{taxonomy_name}_hier_mapper.json"
            )
        else:
            msg.fail(f"Hierarchical mapper not found: {hier_mapper_path}", exits=1)
        # here, let's download the hard-coded taxonomy if it's for escoe
        if taxonomy_name == "esco":
            hard_coded_taxonomy = srsly.read_json(
                data_path / f"hardcoded_ojo_{taxonomy_name}_lookup.json"
            )
            previous_skill_matches = srsly.read_json(
                data_path / f"ojo_{taxonomy_name}_lookup_sample.json"
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
