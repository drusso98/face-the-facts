from llama_index.core.base_retriever import BaseRetriever
from llama_index import QueryBundle


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever, reranker):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)

        # reranking
        reranked_nodes = self.reranker._postprocess_nodes(
            all_nodes,
            query_bundle=QueryBundle(query),
        )
        
#         print(QueryBundle(query).query_str)

        return reranked_nodes
