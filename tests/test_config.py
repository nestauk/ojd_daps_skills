"""
Test cases for configuration managers.
"""

import pytest
from sklearn.pipeline import Pipeline

from ojd_daps_skills import PUBLIC_MODEL_FOLDER_PATH
from ojd_daps_skills.extract_skills.extract_skills_utils import ExtractConfig
from ojd_daps_skills.map_skills.skill_mapper_utils import MapConfig


@pytest.fixture
def extract_config():
    return ExtractConfig.create("nestauk/en_skillner", "nestauk/multiskill-classifier")


@pytest.fixture
def map_config():
    return MapConfig.create("esco")


def test_extract_config(extract_config):
    assert extract_config.ner_model_name == "nestauk/en_skillner"
    assert extract_config.ms_model_name == "nestauk/multiskill-classifier"
    assert isinstance(extract_config.ms_model, Pipeline)

    # assert that there is a multiskill-classifier saved locally
    assert PUBLIC_MODEL_FOLDER_PATH.exists()
    ms_model_path = PUBLIC_MODEL_FOLDER_PATH / "ms_model"

    assert ms_model_path.exists()


# def test_map_config(map_config):
#     assert PUBLIC_DATA_FOLDER_PATH.exists()
#     assert map_config.taxonomy_name == "toy"
#     assert isinstance(map_config.taxonomy_config, dict)
#     assert isinstance(map_config.taxonomy_data, pd.DataFrame)
#     assert isinstance(map_config.hier_mapper, dict)
#     assert isinstance(map_config.taxonomy_embeddings, dict)
