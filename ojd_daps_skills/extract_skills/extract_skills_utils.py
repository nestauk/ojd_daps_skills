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
from spacy.language import Language
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
        nlp (Optional[Language]): The NLP model to use for Named Entity Recognition. This
            is set during creation.
        ms_model (Optional[Pipeline]): The SVM model to use for Multi-Skill classification.
            This is set during creation.
    """

    ner_model_name: str = "nestauk/en_skillner"
    ms_model_name: str = "nestauk/multiskill-classifier"
    nlp: Optional[Language] = None  # Optional, since it's set during creation
    ms_model: Optional[Pipeline] = None  # Optional for the same reason

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(
        cls,
        ner_model_name: Optional[str] = ner_model_name,
        ms_model_name: Optional[str] = ms_model_name,
    ) -> "ExtractConfig":
        """
        Creates an instance of ExtractConfig by loading configurations.

        Parameters:
            ner_model_name (Optional[str]): The name of the NER model to use. Defaults
                to "nestauk/en_skillner".
            ms_model_name (Optional[str]): The name of the Multi-Skill model to use.
                Defaults to "nestauk/multiskill-classifier".

        Returns:
            ExtractConfig: An initialized instance of this configuration class.

        Raises:
            msg.fail: If the models are not loaded locally, this error is raised.
            OSError: If the NER model is not loaded, this error is raised.
        """
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
            if ner_model_name == "nestauk/en_skillner":
                msg.info(f"{ner_model_name} NER model not loaded. Downloading model...")
                os.system(
                    f'pip install "{ner_name} @ https://huggingface.co/{namespace}/{ner_name}/resolve/main/{ner_name}-any-py3-none-any.whl"'
                )
                msg.info("Model downloaded")
                nlp = spacy.load(ner_name)
            else:
                msg.fail(
                    f"{ner_model_name} NER model not loaded: {ner_model_name} Please install accordingly.",
                    exit=1,
                )

        # Load multi-skill model
        ms_model_path = PUBLIC_MODEL_FOLDER_PATH / "ms_model"
        try:
            clf = joblib.load(ms_model_path / "multiskill-classifiert4_v38_0.pkl")
        except Exception:
            msg.fail("Multi-skill classifier not loaded. Downloading model...")
            download(repo_id=ms_model_name, dst=ms_model_path)
            clf = joblib.load(ms_model_path / "multiskill-classifiert4_v38_0.pkl")

        ms_model = Pipeline(
            [("transformer", MultiSkillTransformer()), ("classifier", clf)]
        )

        return cls(
            ner_model_name=ner_model_name,
            ms_model_name=ms_model_name,
            nlp=nlp,
            ms_model=ms_model,
        )
