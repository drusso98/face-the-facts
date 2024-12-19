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


def tokenize_hf(text: List[str], tokenizer) -> Dict[str, torch.Tensor]:
    return tokenizer(text, padding=True, truncation=True, return_tensors='pt')


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def compute_embeddings(texts: List[str], tokenizer, model, disable_tqdm=False):
    # Create an instance of the custom dataset
    max_length = 500  # Adjust as needed
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
            outputs = model(input_ids, attention_mask=attention_mask)

        # Assuming you want to use mean pooling for sentence embeddings
        embedding = mean_pooling(outputs.last_hidden_state, attention_mask)
        embeddings.append(embedding)

    return torch.cat(embeddings, dim=0)


class ContrieverRetriever(BaseRetriever):
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
        self._tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self._model = AutoModel.from_pretrained('facebook/contriever').to('cuda')
        self._similarity_top_k = similarity_top_k
        self._embeddings = compute_embeddings([node.get_content() for node in self._nodes], self._tokenizer, self._model)
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
    ) -> "ContrieverRetriever":
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

        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        return cls(
            nodes=nodes,
            tokenizer = tokenizer,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
        )

    def _get_scored_nodes(self, query: str) -> List[NodeWithScore]:
        query_embedding = compute_embeddings([query], self._tokenizer, self._model, disable_tqdm=True)[0]
        doc_scores = [(query_embedding @ self._embeddings[i]).item() for i in range(len(self._embeddings))]

        nodes: List[NodeWithScore] = []
        for i, node in enumerate(self._nodes):
            nodes.append(NodeWithScore(node=node, score=doc_scores[i]))

        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:

        scored_nodes = self._get_scored_nodes(query_bundle.query_str)

        # Sort and get top_k nodes, score range => 0..1, closer to 1 means more relevant
        nodes = sorted(scored_nodes, key=lambda x: x.score or 0.0, reverse=True)
        return nodes[: self._similarity_top_k]
