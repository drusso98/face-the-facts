input_file="generation/models_list.txt"

# Loop through each line of the file
while IFS=, read -r model_id model_type quantization; do
    # Do something with the elements
    echo "Element 1: $model_id"
    echo "Element 2: $model_type"
    echo "Element 3: $quantization"
    python generation/generation_code_chunks.py -model_id $model_id -model_type $model_type -quantization $quantization
    python generation/generation_code_articles.py -model_id $model_id -model_type $model_type -quantization $quantization
done < "$input_file"
python evaluation/automatic_eval.py -input_dir results/test_set/chunks -output_dir evaluation/results/automatic_eval/chunks -context_len 10
python evaluation/automatic_eval.py -input_dir results/test_set/articles -output_dir evaluation/results/automatic_eval/articles -context_len 1
