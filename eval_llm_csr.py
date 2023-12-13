import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def load_model(model_name, bnb_config, max_memory_cap):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{max_memory_cap}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def predict_result(question, options, tokenizer, model, answer_offset, context=None):
    if context is not None:
        prompt = context + " " + question
    else:
        prompt = question
    prompts = []
    model.eval()
    min_pp = float('inf')
    res = -1
    for idx, option in enumerate(options):
        current_prompt = prompt.rstrip(".") + " " + option
        with torch.no_grad():
            input_ids = tokenizer(current_prompt, return_tensors='pt', padding=True, truncation=True)['input_ids'].to('cuda')
            loss = model(input_ids, labels=input_ids).loss
        perplexity = torch.exp(loss).detach().cpu().item()
        if perplexity < min_pp:
            min_pp = perplexity
            res = idx+answer_offset
    return res

def main(args):
    model_name, max_memory_cap = args.model_name, args.max_memory_cap
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config, max_memory_cap)
    test_file, context, question, options, answer = args.test_file, args.context, args.question, eval(args.options), args.answer
    lst = []
    with open(test_file, "r") as file:
        lst = json.load(file)
    matches = []
    for instance in tqdm(lst):
        answer_val = int(instance[answer])
        context_val = None
        if context is not None:
            context_val = instance[context]
        question_val = instance[question]
        options_val = []
        for option in options:
            options_val.append(instance[option])
        pred = predict_result(question_val, options_val, tokenizer, model, answer_val, context_val)
        matches.append(pred==answer_val)
    print(f"Accuracy: {sum(matches)/len(matches)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--max_memory_cap', type=int, default=8192)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--context', type=str, default=None)
    parser.add_argument('--question', type=str)
    parser.add_argument('--options', type=str)
    parser.add_argument('--answer', type=str)

    args = parser.parse_args()
    main(args)