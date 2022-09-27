from sentence_transformers import SentenceTransformer
import time
from ojd_daps_skills import logger
import logging


class BertVectorizer:
    """
    Use a pretrained transformers model to embed sentences.
    In this form so it can be used as a step in the pipeline.
    """

    def __init__(
        self,
        bert_model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
        verbose=True,
    ):
        self.bert_model_name = bert_model_name
        self.batch_size = batch_size
        self.verbose = verbose
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.ERROR)

    def fit(self, *_):
        self.bert_model = SentenceTransformer(self.bert_model_name)
        self.bert_model.max_seq_length = 512
        return self

    def transform(self, texts):
        logger.info(f"Getting embeddings for {len(texts)} texts ...")
        t0 = time.time()
        sel.embedded_x = self.bert_model.encode(texts, batch_size=self.batch_size)
        logger.info(f"Took {time.time() - t0} seconds")
        return self.embedded_x
