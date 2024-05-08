from typing import List, Union

from pydantic import BaseModel
from spacy.tokens import Doc
from wasabi import msg

from ojd_daps_skills import setup_spacy_extensions
from ojd_daps_skills.extract_skills.extract_skills_utils import ExtractConfig
from ojd_daps_skills.extract_skills.multiskill_rules import (
    _split_duplicate_object, _split_duplicate_verb, _split_skill_mentions)
from ojd_daps_skills.map_skills.skill_mapper import SkillsMapper
from ojd_daps_skills.utils.text_cleaning import clean_text, short_hash

setup_spacy_extensions()


class SkillsExtractor(BaseModel):
    """
    SkillsExtractor class to EXTRACT and MAP skills from job ads.

    Attributes:
        ner_model_name (str): spaCy NER model name to load from Hugging Face.
        ms_model_name (str): multiskill model name to load from Hugging Face.
        taxonomy_name (str): pre-defined skills taxonomy name to load data for.
    """

    taxonomy_name: str = "toy"
    ner_model_name: str = "nestauk/en_skillner"
    ms_model_name: str = "nestauk/multiskill-classifier"

    def __init__(
        self,
    ):
        super().__init__()
        self._extract_config: ExtractConfig = ExtractConfig.create(
            ner_model_name=self.ner_model_name,
            ms_model_name=self.ms_model_name,
        )
        self._skill_mapper: SkillsMapper = SkillsMapper(
            taxonomy_name=self.taxonomy_name
        )

    def extract_skills(self, job_ads: Union[str, List[str]]) -> List[Doc]:
        """Return a list of spaCy Doc objects with entities
            and ._.skill_spans attribute that includes all
            'SKILL' spans, including multiskill and split
            multiskill ones. To access the original entities
            including 'SKILL', 'BENEFIT' and 'EXPERIENCE',
            use the .ents attribute.

        Args:
            job_ads (Union[str, List[str]]): single or list of job ads.

        Returns:
            Union[Doc, List[Doc]]: single or list of spaCy Doc objects
                with ._.skill_spans attribute.
        """
        if isinstance(job_ads, str):
            return [self.get_skills(job_ads)]

        elif not isinstance(job_ads, list) or not all(
            isinstance(ad, str) for ad in job_ads
        ):
            raise msg.fail(
                "Input must be a string or a list of strings containing job ad texts.",
                exits=1,
            )

        return [self.get_skills(job_ad) for job_ad in job_ads]

    # map skills function

    def get_skills(self, job_ad: str) -> Doc:
        """Return a spaCy Doc object with entities
            and split 'SKILL' spans.

        Args:
            job_ad (str): job ad text.

        Returns:
            Doc: spaCy Doc object with split 'SKILL' spans.
        """
        rules = [_split_duplicate_object, _split_duplicate_verb, _split_skill_mentions]

        job_ad_clean = clean_text(job_ad)
        doc = self._extract_config.nlp(job_ad_clean)

        all_skill_ents = []
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                ms_pred = self._extract_config.ms_model.predict([ent.text])[0]
                if ms_pred == 1:
                    for rule in rules:
                        split_ent = rule(ent)
                        if split_ent:
                            all_skill_ents.append(split_ent)
                    # else, if no split, append the original entity
                    all_skill_ents.append(ent)
                else:
                    all_skill_ents.append(ent)

        doc._.skill_spans = all_skill_ents

        return doc

    def map_skills(self, job_ads: Union[Doc, list[Doc]]) -> List[Doc]:
        """Return a list of spaCy Doc objects with entities
            and ._.mapped_skills attribute that includes all
            mapped skills.

        Args:
            doc (Union[Doc, List[Doc]]): single or list of spaCy Doc objects
                with ._.skill_spans attribute.

        Returns:
            Union[Doc, List[Doc]]: single or list of spaCy Doc objects
                with ._.mapped_skills attribute.
        """
        if isinstance(job_ads, Doc):
            job_ads = [job_ads]

        elif not isinstance(job_ads, list) or not all(
            isinstance(doc, Doc) for doc in job_ads
        ):
            raise msg.fail(
                "Input must be a spaCy Doc object or a list of spaCy Doc objects with ._.skill_spans attribute.",
                exits=1,
            )

        all_mapped_skills = self._skill_mapper.match_skills(job_ads)

        for job_ad in job_ads:
            mapped_skills_list = []
            for skill_span in job_ad._.skill_spans:
                if not isinstance(skill_span, str):
                    skill_span = skill_span.text
                skill_hash = short_hash(skill_span)
                mapped_skills_list.append(all_mapped_skills.get(skill_hash))

            job_ad._.mapped_skills = mapped_skills_list

        return job_ads

    def __call__(self, job_ads: Union[str, List[str]]) -> List[Doc]:
        doc = self.extract_skills(job_ads)
        doc = self.map_skills(doc)

        return doc
