import pandas as pd
import torch
from tqdm import tqdm
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import os
import random
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_id", type=str, required=True, help="hugginface model id")
    parser.add_argument("-model_type", type=str, required=True, help="family name of the model",
                        choices=['mistral', 'llama2','llama3'])
    parser.add_argument("-quantization", type=str, required=True, help="q: load the quantized, nq: not quantized",
                       choices=['q','nq'])
    return parser.parse_args()


def get_data(retrieved_data):
    data = {}
    for idx, row in retrieved_data.iterrows():
        data[idx] = {}
        data[idx]['context'] = [el['text'] for el in row['nodes']]
        data[idx]['query'] = row['query']
    return data


def generate_prompts(instruction, data, randomize=False, context_num=10, model_name = '', sys_prompt=''):
    prompts = []
    for idx,item in data.items():
        if randomize:
            random.seed(2024)
            random.shuffle(item['context'][:context_num])
        context_str = "\n\n".join(item['context'][:context_num])
        final_inst = instruction.format(context_str=context_str, query_str=item['query'])
        if model_name == 'llama3':
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": final_inst},
            ]
            prompts.append(messages)
        elif model_name == 'llama2':
            prompts.append(f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n{final_inst} [/INST]")
        elif model_name == 'mistral':
            prompts.append(f"[INST] {sys_prompt} {final_inst} [/INST]")         
    return prompts


def generate(text, model, model_type, tokenizer):
    if model_type == 'llama3':
        input_ids = tokenizer.apply_chat_template(
            text,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=500,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    else:
        input_ids = tokenizer([text], return_tensors="pt").to('cuda')
        outputs = model.generate(**input_ids, max_new_tokens=300, do_sample=True)
        response = tokenizer.batch_decode(outputs)[0]
    return response


args = parse_arguments()
print(args)
model_id = args.model_id
model_type = args.model_type
quantization = bool(args.quantization == "q")

#load the data
print(">>> loading data...\n")
folder_path = "path_retrieved_data"
retrieved_data = [(el, pd.read_json(folder_path+el).T) for el in os.listdir(folder_path) if "chunks" in el]

with open("sys_prompt.txt", "r") as f:
    system_prompt = f.read()

with open("instruction.txt", "r") as f:
    instruction_template = f.read()

print("HUGGINGFACE LOGIN")
login("<INSERT_HF_TOKEN>")

#load tokenizer
print('>>> loading tokenizer...\n')
tokenizer = AutoTokenizer.from_pretrained(model_id)

#load the model
print('>>> loading the model...\n')
if quantization:
    # quantization
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
    )

else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


for file_name, configuration in retrieved_data:
    # generating prompts
    data = get_data(configuration)
    prompts = generate_prompts(instruction_template, data, randomize=False, model_name=model_type, sys_prompt=system_prompt)
    generations = [generate(instruction, model, model_type, tokenizer) for instruction in tqdm(prompts)]
    replies = [g.split('[/INST]')[1].strip('</s> ') if model_type != 'llama3' else g for g in generations]

    for idx,(el,prompt) in enumerate(zip(replies,prompts)):
        data[idx]['reply'] = el
        data[idx]['prompt'] = prompt

    df = pd.DataFrame(data)
    # save generations
    final_file_name = model_id.split("/")[1]+"_"+file_name
    df.to_json(f"results/extra/zero{final_file_name}", indent=4)