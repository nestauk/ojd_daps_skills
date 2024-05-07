from typing import Any, List, Union

from .extract_skills_utils import ExtractConfig, MapConfig
from .multiskill_rules import (
    _split_duplicate_object,
    _split_duplicate_verb,
    _split_skill_mentions,
)
from ..map_skills.skill_ner_mapper import SkillMapper
from ..utils.text_cleaning import clean_text, short_hash

from pydantic import BaseModel
from spacy.tokens import Doc
from wasabi import msg


class SkillsExtractor(BaseModel):
    """
    SkillsExtractor class to extract and map skills from job ads.

    Attributes:
        extract_config (ExtractConfig): ExtractConfig configuration manager to load
            custom spaCy NER model and multiskill models.
        map_config (MapConfig): MapConfig configuration manager to load relevant
            skill mapping data.
        skill_mapper (SkillMapper): SkillMapper object to map extracted skills
            onto a pre-defined skills taxonomy.
    """

    extract_config: ExtractConfig
    map_config: MapConfig
    skill_mapper: SkillMapper = SkillMapper(config=map_config)

    def extract_skills(self, job_ads: Union[str, List[str]]) -> Union[Doc, List[Doc]]:
        """Return a list of spaCy Doc objects with entities
            and ._.skill_spans attribute that includes all
            'SKILL' spans, including split ones.

        Args:
            job_ads (Union[str, List[str]]): single or list of job ads.

        Returns:
            Union[Doc, List[Doc]]: single or list of spaCy Doc objects
                with ._.skill_spans attribute.
        """
        if isinstance(job_ads, str):
            return self.get_skills(job_ads)

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
            job_ad (JobAd): JobAd object with job_ad attribute.

        Returns:
            Doc: spaCy Doc object with split 'SKILL' spans.
        """
        rules = [_split_duplicate_object, _split_duplicate_verb, _split_skill_mentions]

        job_ad_clean = clean_text(job_ad)
        doc = self.extract_config.nlp(job_ad_clean)

        all_skill_ents = []
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                ms_pred = self.extract_config.ms_model.predict([ent.text])[0]
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

        if not Doc.has_extension("mapped_skills"):
            Doc.set_extension("mapped_skills", default=[], force=True)

        all_mapped_skills = self.skill_mapper.match_skills(job_ads)

        for job_ad in job_ads:
            mapped_skills_list = []
            for skill_span in job_ad._.skill_spans:
                skill_hash = short_hash(skill_span.text)
                mapped_skills_list.append(all_mapped_skills.get(skill_hash))

            job_ad._.mapped_skills = mapped_skills_list

        return job_ads

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Extract and map skills from job ads.
        """
        docs = self.extract_skills(*args, **kwds)
        docs = self.map_skills(docs)

        return docs
