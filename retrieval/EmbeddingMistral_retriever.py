import logging
from typing import Callable, List, Optional, cast, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tqdm
# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
from llama_index.callbacks.base import CallbackManager
from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.base_retriever import BaseRetriever
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.storage.docstore.types import BaseDocumentStore

logger = logging.getLogger(__name__)


class E5MistralRetriever(BaseRetriever):
    def __init__(
        self,
        nodes: List[BaseNode],
        tokenizer: Any,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:

        self._nodes = nodes
        self._model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
        self._similarity_top_k = similarity_top_k
        self._ctx_embeddings = np.vstack([self._model.encode(node.get_content()) for node in self._nodes])

        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @classmethod
    def from_defaults(
        cls,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        tokenizer: Any = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
    ) -> "E5MistralRetriever":
        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert (
            nodes is not None
        ), "Please pass exactly one of index, nodes, or docstore."

        return cls(
            nodes=nodes,
            tokenizer=tokenizer,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
        )

    def _get_scored_nodes(self, query: str) -> List[NodeWithScore]:
        prompt = "Instruct: Retrieve relevant documents to support or refute the given claim.\nQuery: "
        query_embedding = self._model.encode(query, prompt=prompt)
        scores = (query_embedding @ self._ctx_embeddings.T) * 100
        doc_scores = scores.tolist()
        nodes: List[NodeWithScore] = []
        for i, node in enumerate(self._nodes):
            nodes.append(NodeWithScore(node=node, score=doc_scores[i]))

        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:

        scored_nodes = self._get_scored_nodes(query_bundle.query_str)

        # Sort and get top_k nodes, score range => 0..1, closer to 1 means more relevant
        nodes = sorted(scored_nodes, key=lambda x: x.score or 0.0, reverse=True)
        return nodes[: self._similarity_top_k]
