import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.retrievers import BM25Retriever
from bm25plus_retriever import BM25PlusRetriever
from contriever_retriever import ContrieverRetriever
from dragon_retriever import DragonRetriever
from hybrid_retriever import HybridRetriever
from EmbeddingMistral_retriever import E5MistralRetriever
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.schema import Document
from llama_index.embeddings import HuggingFaceEmbedding

from ranx import Qrels, Run, evaluate

import spacy
import pandas as pd
from tqdm import tqdm


def load_data(queries, documents):
    with open(queries, "r") as f:
        queries = f.read().split('\n')
        queries = [query.strip() for query in queries]

    with open(documents, "r") as f:
        documents = f.read().split('\n')
        documents = [document.strip() for document in documents]

    return queries, documents


def chunking(documents,chunk_size, llm, embed_model, sentences=False, semantic=False):

    if sentences:  # sentence splitting if required
        nlp = spacy.load("en_core_web_md")
        documents = [nlp(doc) for doc in tqdm(documents)]

        docs = []
        for i, doc in enumerate(documents):
            for sent in doc.sents:
                d = Document(text=sent.text)
                d.metadata['file_name'] = f'doc_{i}.txt'
                docs.append(d)
    else:
        docs = []
        for i, doc in enumerate(documents):
            d = Document(text=doc)
            d.metadata['file_name'] = f'doc_{i}.txt'
            docs.append(d)

    # chunking and nodes creation
    if len(docs) > 0:
        if semantic:
            splitter = SemanticSplitterNodeParser(
                buffer_size=1, breakpoint_percentile_threshold=90, embed_model=embed_model
            )
        else:
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0)
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, transformations=[splitter])
        nodes = service_context.node_parser.get_nodes_from_documents(docs)
    else:
        print("No documents found")
        sys.exit(1)
    return nodes, service_context


def indexing(nodes, service_context):
    # initialize storage context (by default it's in-memory)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True
    )
    return index


def set_retriever(nodes, top_k, retriever):
    if retriever == 'BM25':
        retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k,
                                                tokenizer=lambda text: str.split(text))
    elif retriever == 'BM25+':
        retriever = BM25PlusRetriever.from_defaults(nodes=nodes, similarity_top_k=top_k,
                                                tokenizer=lambda text: str.split(text))
    elif retriever == 'contriever':
        retriever = ContrieverRetriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
    elif retriever == 'dragon':
        retriever = DragonRetriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
    elif retriever == 'mistral':
        retriever = E5MistralRetriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
    else:
        print("Choose a valid retriever. Options: BM25, BM25+, contriever, dragon, mistral")
    return retriever


def set_hybrid_retriever(retriever_1, retriever_2, reranker):
    return HybridRetriever(retriever_1, retriever_2, reranker)


def set_reranker(k, model="BAAI/bge-reranker-large"):
    return SentenceTransformerRerank(top_n=k, model=model)


def single_retrieve(retriever, query):
    return retriever.retrieve(query)


def create_eval_dataset(results, nodes):
    # qrels_dict = Dict{query_id:{doc_id:score}}
    qrels_dict = {}
    for node in nodes:
        if f'q_{int(node.metadata["file_name"].strip("doc_.txt"))}' in qrels_dict:
            qrels_dict[f'q_{int(node.metadata["file_name"].strip("doc_.txt"))}'].update({node.id_:1})
        else:
            qrels_dict[f'q_{int(node.metadata["file_name"].strip("doc_.txt"))}'] = {node.id_:1}
    # run_dict = Dict{query_id:{retriever_doc_id:score}}
    run_dict = {f'q_{idx}':{node.id_:node.score for node in query} for idx,query in enumerate(results)}
    return qrels_dict,run_dict


def evaluate_dataset(qrels_dict,run_dict,run_name,k):
    eval_metric = ["mrr@{k}", "hit_rate@{k}", "precision@{k}", "recall@{k}", "f1@{k}", "map@{k}"]
    runs = [metric.format(k=i) for i in range(1,11) for metric in eval_metric]
    runs.sort()
    qrels = Qrels(qrels_dict)
    run = Run(run_dict, name=run_name)
    eval_results = evaluate(qrels, run, runs)
    return eval_results


def display_results(name, eval_results, save=False, save_path=None):
    df = pd.DataFrame(eval_results, index=[name])

    if save:
        df.to_json('{}/{}.json'.format(save_path,name), orient="records")

    return df


def set_embed_model(model_name, pooling):
    if model_name is None:
        return None
    else:
        return HuggingFaceEmbedding(model_name=model_name, pooling=pooling)


def retrieve_and_save(retriever, queries, save=False, save_path=""):
    retrieved_documents = [retriever.retrieve(q) for q in tqdm(queries)]
    data = {}
    for i, (r, q) in enumerate(zip(retrieved_documents, queries)):
        data[i] = {'query': q}
        extracted_nodes = []
        for n in r:
            extracted_nodes.append({'text': n.text,
                                    'similarity': n.score,
                                    'file_name': n.metadata['file_name']})
        data[i]['nodes'] = extracted_nodes

    if save:
        pd.DataFrame(data).to_json('{}.json'.format(save_path), indent=2)

    return data
