import logging
from typing import Callable, List, Optional, cast, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from transformers import AutoTokenizer, AutoModel
from llama_index.callbacks.base import CallbackManager
from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.base_retriever import BaseRetriever
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.storage.docstore.types import BaseDocumentStore

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze()}


def tokenize_hf(text: List[str]) -> Dict[str, torch.Tensor]:
#     tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
    return self_.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to('cuda')


def compute_embeddings(texts: List[str], tokenizer, model, disable_tqdm=False):
    # Create an instance of the custom dataset
    max_length = 300  # Adjust as needed
    text_dataset = TextDataset(texts, tokenizer, max_length)

    # Create a DataLoader to handle batches
    batch_size = 2  # Adjust as needed
    dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Process batches
    embeddings = []

    for batch in tqdm.tqdm(dataloader, disable=disable_tqdm):
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        embeddings.append(outputs)

    return torch.cat(embeddings, dim=0)


class DragonRetriever(BaseRetriever):
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
        self._tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
        self._query_encoder = AutoModel.from_pretrained('facebook/dragon-plus-query-encoder').to('cuda')
        self._context_encoder = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder').to('cuda')
        self._similarity_top_k = similarity_top_k
        self._ctx_embeddings = compute_embeddings([node.get_content() for node in self._nodes],
                                              self._tokenizer, self._context_encoder)

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
    ) -> "DragonRetriever":
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

        tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
        return cls(
            nodes=nodes,
            tokenizer=tokenizer,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
        )

    def _get_scored_nodes(self, query: str) -> List[NodeWithScore]:        
        query_embedding = compute_embeddings([query], self._tokenizer, self._query_encoder, disable_tqdm=True)[0]
        doc_scores = [(query_embedding @ self._ctx_embeddings[i]).item() for i in range(len(self._ctx_embeddings))]
        
        nodes: List[NodeWithScore] = []
        for i, node in enumerate(self._nodes):
            nodes.append(NodeWithScore(node=node, score=doc_scores[i]))

        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:

        scored_nodes = self._get_scored_nodes(query_bundle.query_str)

        # Sort and get top_k nodes, score range => 0..1, closer to 1 means more relevant
        nodes = sorted(scored_nodes, key=lambda x: x.score or 0.0, reverse=True)
        return nodes[: self._similarity_top_k]
