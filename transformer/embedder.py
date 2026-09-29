"""Pretrained word vectors projected into a small, consistent coordinate space."""

import json
import string
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from sklearn.decomposition import PCA


class TokenEmbedder:
    """Look up GloVe vectors and reduce their dimensions without model training.

    PCA is fitted once per instance on a fixed reference vocabulary, never on
    the input sentence. Unknown words map to the zero vector in output space.
    """

    MODEL_ID = "sentence-transformers/average_word_embeddings_glove.6B.300d"
    CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"

    def __init__(self, dimensions: int = 3) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise ValueError("dimensions must be an integer between 1 and 300")
        if not 1 <= dimensions <= 300:
            raise ValueError("dimensions must be an integer between 1 and 300")
        self.dimensions = dimensions

        tokenizer_path = self._download("whitespacetokenizer_config.json")
        with open(tokenizer_path, encoding="utf-8") as source:
            vocabulary = json.load(source)["vocab"]
        self._word_ids = {word: index for index, word in enumerate(vocabulary)}

        weights_path = self._download("model.safetensors")
        # The first row is the unknown/padding vector. Use the next 10,000
        # vocabulary entries as a fixed reference set for a manageable PCA.
        with safe_open(weights_path, framework="numpy") as weights:
            self._vectors = weights.get_tensor("emb_layer.weight")
        self._projection = PCA(n_components=dimensions, svd_solver="full")
        self._projection.fit(self._vectors[1:10001])

    def _download(self, filename: str) -> str:
        return hf_hub_download(
            repo_id=self.MODEL_ID,
            filename=f"0_WordEmbeddings/{filename}",
            cache_dir=str(self.CACHE_DIR),
        )

    def embed(self, tokens: list[str]) -> np.ndarray:
        """Return one vector per token, with shape (len(tokens), dimensions).

        Lookup is lowercase with surrounding ASCII punctuation removed.
        Tokens themselves are unchanged; repeated words have identical vectors.
        """
        result = np.zeros((len(tokens), self.dimensions), dtype=np.float32)
        positions = []
        word_ids = []
        for position, token in enumerate(tokens):
            word = token.lower().strip(string.punctuation)
            word_id = self._word_ids.get(word)
            if word_id is not None and word_id != 0:
                positions.append(position)
                word_ids.append(word_id)
        if word_ids:
            result[positions] = self._projection.transform(self._vectors[word_ids])
        return result
