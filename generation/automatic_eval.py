import os
import pandas as pd
from tqdm import tqdm
from string2string.similarity import BARTScore
from string2string.metrics import ROUGE
from summac.model_summac import SummaCConv
import numpy as np
import nltk
nltk.download('punkt')
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", type=str, required=True, help="dir of the input data")
    parser.add_argument("-output_dir", type=str, required=True, help="name for output file")
    parser.add_argument("-context_len", type=int, required=True, help="# of nodes from the context")
    return parser.parse_args()


def preprocess_data(file):
    context, claims, answers = ([] for i in range(3))
    df = pd.read_json(f"{input_dir}/{file}").T
    context = df.context.to_list()
    claims = df['query'].to_list()
    answers = [el.strip('Reply: \n') for el in df['reply'].to_list()]
    return context, claims, answers


def compute_ROUGE(answers, context):
    # Let's define a list of candidate sentences and a list of reference sentences
    candidates = answers
    references = [["\n".join(chunk[:context_len])] for chunk in context]
    result = rogue.compute(candidates, references)
    return result


def compute_BARTScore(answers, context):
    # Define the source and target sentences
    source_sentences = [" ".join(el[:context_len]) for el in context]
    target_sentences = answers
    # Compute the BARTScore
    score = bart_scorer.compute(source_sentences, target_sentences, agg="mean", batch_size=4)
    return score['score'].mean()


def compute_verdict_sim(gold, generated):
    # Define the source and target sentences
    source_sentences = gold
    target_sentences = generated
    # Compute the BARTScore
    score = bart_scorer.compute(source_sentences, target_sentences, agg="mean", batch_size=4)
    return score['score'].mean()


def compute_summac(answers, context):
    documents = ["\n".join(c[:context_len]) for c in context]
    score_conv1 = model_conv.score(documents, answers)
    return score_conv1['scores']


args = parse_arguments()
print(args)
input_dir = args.input_dir
output_dir = args.output_dir
context_len = args.context_len

# Initializing metrics
rogue = ROUGE()
bart_scorer = BARTScore(model_name_or_path='facebook/bart-large-cnn', device='cuda')
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")


# loading datasets for verdicts
fullfact_verdicts = pd.read_json("../../data/fullfact/fullfact_test.json").T['VERDICT'].to_list()
vermouth = pd.read_excel("../../data/VerMouth/vermouth_article_test.xlsx")

files = os.listdir(input_dir)
d = {}
for file in tqdm(files):
    context, claims, answers = preprocess_data(file)
    # getting the verdicts
    dataset = file.split("_")[1]
    if dataset == 'fullfact':
        verdicts = fullfact_verdicts
    else: 
        verdicts = vermouth[vermouth.label == dataset]['tweet_verdict'].to_list()
    
    d[file] = compute_ROUGE(answers, context)
    d[file]['BARTScore'] = compute_BARTScore(answers, context)
    d[file]['summac'] = compute_summac(answers, context)
    d[file]['gold_similarity'] = compute_verdict_sim(verdicts, answers)


res_df = pd.DataFrame(d).T
res_df.sort_index()
res_df['summac'] = res_df.summac.apply(lambda x:sum(x)/len(x))
res_df = res_df.astype(float).round(3)
res_df.sort_index(inplace=True)

mapping = {'Llama-2-7b-chat-hf':'llama2-7b',
           'Llama-2-13b-chat-hf': 'llama2-13b',
           'Mistral-7B-Instruct-v0.1': 'mistral-7b-v0.1',
           'Mistral-7B-Instruct-v0.2': 'mistral-7b-v0.2',
           'Meta-Llama-3-8B-Instruct': 'llama3-8b'}

res_df['dataset'] = res_df.index.map(lambda x:x.split('_')[1])
res_df['extra_evidence'] = res_df.index.map(lambda x:"extra" in x)
res_df.index = res_df.index.map(lambda x:mapping[x.split('_')[0]])
res_df.index.name = "model"
res_df.reset_index(inplace=True)

res_df.to_excel(f"{output_dir}/automatic_scores.xlsx", index=False)