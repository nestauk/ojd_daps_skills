import time

from sentence_transformers import SentenceTransformer
from wasabi import msg


class BertVectorizer:
    """
    BertVectorizer class to embed sentences using a pretrained transformers model.

    Attributes:
        bert_model_name (str): The name of the pretrained model to use from HuggingFace Hub.
        multi_process (bool): Whether to use multiprocessing for embedding.
        batch_size (int): The batch size to use for embedding.
        verbose (bool): Whether to log info messages.
    """

    def __init__(
        self,
        bert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        multi_process: bool = False,
        batch_size: int = 32,
    ):
        self.bert_model_name = bert_model_name
        self.multi_process = multi_process
        self.batch_size = batch_size

    def fit(self, *_):
        self.bert_model = SentenceTransformer(self.bert_model_name, device="cpu")
        self.bert_model.max_seq_length = 512
        return self

    def transform(self, texts):
        msg.info(f"Getting embeddings for {len(texts)} texts ...")
        t0 = time.time()
        if self.multi_process:
            msg.info(".. with multiprocessing")
            pool = self.bert_model.start_multi_process_pool()
            self.embedded_x = self.bert_model.encode_multi_process(
                texts, pool, batch_size=self.batch_size
            )
            self.bert_model.stop_multi_process_pool(pool)
        else:
            self.embedded_x = self.bert_model.encode(texts, batch_size=self.batch_size)
        msg.info(f"Took {time.time() - t0} seconds")
        return self.embedded_x
