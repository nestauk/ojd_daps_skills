"""
Utility functions and a configuration manager for 
extracting skills from job descriptions.
"""

import os
from typing import List, Optional

import joblib
import spacy
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from skops.hub_utils import download
from spacy.tokens import Doc
from wasabi import msg

from ojd_daps_skills import PUBLIC_MODEL_FOLDER_PATH


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
        ner_model_name (str): The name of the Named Entity Recognition model to
            use from HuggingFace Hub. Current configuration supports "nestauk/en_skillner".
            You can use your own NER model if you have a custom NER model to extract skills.
        ms_model_name (str): The name of the Multi-Skill model to use. Current configurations
            supports "nestauk/multiskill-classifier".
        nlp (spacy.Language): spaCy NLP model.
        ms_model (Pipeline): Multi-Skill model pipeline.
    """

    ner_model_name: str = "nestauk/en_skillner"
    ms_model_name: str = "nestauk/multiskill-classifier"
    nlp: spacy.Language
    ms_model: Pipeline

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(
        cls, ner_model_name: Optional[str] = None, ms_model_name: Optional[str] = None
    ) -> "ExtractConfig":
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
        # Use default values if none provided
        ner_model_name = ner_model_name or cls.ner_model_name
        ms_model_name = ms_model_name or cls.ms_model_name

        Doc.set_extension("skill_spans", default=[], force=True)

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
            nlp = spacy.load(ner_name)

        # Load multi-skill model
        ms_model_path = PUBLIC_MODEL_FOLDER_PATH / "ms_model"
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
