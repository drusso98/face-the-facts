import argparse
import utils


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-queries", type=str, required=True, help="path to queries file")
    parser.add_argument("-documents", type=str, required=True, help="path to documents file")
    parser.add_argument("-chunk_size", type=int, default=200, help="chunk size for document splitting")
    parser.add_argument("-sentences", action='store_true', help="for sentence retrieval")
    parser.add_argument("-top_k", type=int, required=True, help="number of chunks to retrieve")
    parser.add_argument("-retriever", type=str, required=False, default='BM25',
                        choices=['BM25', 'BM25+', 'contriever', 'dragon', 'mistral'])
    parser.add_argument("-save", action='store_true', help="True for saving results")
    parser.add_argument("-save_path", type=str, required=False, default=None, help="path to save results")
    parser.add_argument("-save_name", type=str, required=False, default="scores", help="name of the file")
    return parser.parse_args()


def main(args):
    print(">> experiment is starting")
    print(f">> loading queries from: {args.queries} and documents from: {args.documents}")
    queries, documents = utils.load_data(args.queries, args.documents)
    print(f">> loaded {len(queries)} queries and {len(documents)} documents")
    embed_model = None
    llm = None
    print(f">> chunking...")
    nodes, service_context = utils.chunking(documents, args.chunk_size, llm, embed_model, sentences=args.sentences)
    print(f">> number of nodes: {len(nodes)}")
    print(f">> setting the retriever {args.retriever} (top_k {args.top_k})")
    retriever = utils.set_retriever(nodes, args.top_k, args.retriever)
    print(f">> retrieving and saving results in {args.save_path}")
    utils.retrieve_and_save(retriever, queries, save=args.save, save_path=args.save_path)


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
