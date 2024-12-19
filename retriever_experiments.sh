for style in "fullfact" "SMP" "anger" "disgust" "fear" "happiness" "sadness" "surprise"
do
    echo "style == $style"
    for retriever in "BM25" "BM25+" "dragon" "contriever" "mistral"
    do
        echo "retriever == $retriever"
        python retrieval/retrieval_experiments.py -queries path_to_claims -documents path_to_articles -chunk-size 100 -top_k 10 -retriever ${retriever} -save -save_path your_save_path_here -save_name ${retriever}_chunks_norep_${style}
        python retrieval/retrieval_experiments.py -queries path_to_claims -documents path_to_articles -chunk-size 5000 -top_k 10 -retriever ${retriever} -save -save_path your_save_path_here -save_name ${retriever}_articles_norep_${style}
    done
    for i in 1 2 3 4 5 6 7 8 9 10
    do
        echo "k == $i"
        python retrieval/hybrid_retrieval_experiments.py -queries path_to_claims -documents path_to_articles -chunk-size 100 -top_k $i -top_k_reranker $i -retriever1 BM25+ -retriever2 dragon -save -save_path your_save_path_here -save_name hybrid_chunks_norep_${style}_${i}
        python retrieval/hybrid_retrieval_experiments.py -queries path_to_claims -documents path_to_articles -chunk-size 5000 -top_k $i -top_k_reranker $i -retriever1 BM25+ -retriever2 dragon -save -save_path your_save_path_here -save_name hybrid_articles_norep_${style}_${i}
    done
done
