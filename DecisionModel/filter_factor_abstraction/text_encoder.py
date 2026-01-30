"""
冻结的文本编码器：sentence-level encoder 或 CLIP text encoder，将自然语言筛选要素编码为向量。
"""

from __future__ import annotations

from typing import List, Union

import numpy as np


class TextEncoder:
    """
    使用冻结的 sentence-transformers 或 CLIP 文本编码器，将文本列表编码为 (N, D) 向量矩阵。
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        if "clip" in self.model_name.lower():
            self._load_clip_text()
        else:
            self._load_sentence_transformer()

    def _load_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # 冻结
            for p in self._model.parameters():
                p.requires_grad = False
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for sentence-level encoder. "
                "Install with: pip install sentence-transformers"
            )

    def _load_clip_text(self) -> None:
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad = False
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to(self.device)
        except ImportError:
            raise ImportError(
                "transformers and torch are required for CLIP. "
                "Install with: pip install transformers torch"
            )

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表编码为 (N, D) 的 numpy 数组，L2 归一化便于后续余弦相似度。
        """
        self._lazy_load()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if "clip" in self.model_name.lower():
            return self._encode_clip(texts)
        return self._encode_sentence_transformer(texts)

    def _encode_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        emb = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def _encode_clip(self, texts: List[str]) -> np.ndarray:
        import torch
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
        out = out.cpu().numpy()
        # L2 归一化
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        out = out / norm
        return out.astype(np.float32)

    @property
    def embed_dim(self) -> int:
        self._lazy_load()
        if "clip" in self.model_name.lower():
            return self._model.config.text_config.hidden_size
        return self._model.get_sentence_embedding_dimension()
