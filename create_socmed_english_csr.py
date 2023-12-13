
import argparse
from datasets import load_dataset
import os
from openai import OpenAI
from tqdm import tqdm
import re
import json


SYSTEM_INSTRUCTION = "You are a young English speaker."


LLM_TYPE = "gpt-4"
MAX_TOKEN_SIZE = 8192

OPTIONS_SEP = ";"
SECTION_SEP = ":"

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model(LLM_TYPE)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def mask_answer(txt, answer):
    txt = txt.replace("Translation: ","")
    print(txt)
    splitted = txt.split(answer)
    print(splitted)
    return f"{splitted[0]} {answer} {splitted[1]} _ {splitted[2]}"

def get_translated_siqa_fields_from_response(response, df):
    choices = response.choices
    siqa_like = []
    for choice in tqdm(choices, position=0, leave=True):
        msg = choice.message.content
        text_sep = r'<sep>\n|<sep>'
        examples = re.split(text_sep, msg)
        for example in tqdm(examples, position=0, leave=True):
            print(example)
            siqa_like.append(example)
    return siqa_like

def get_copa_prefix(question_type, is_formal):
    if not is_formal:
        return 'cuz' if question_type=='cause' else ', so'
    return 'because' if question_type=='cause' else ', so'

def get_translated_copa_fields_from_response(response, selected_field, df):
    choices = response.choices
    
    copa_like = []
    idx = 0
    for choice in tqdm(choices, position=0, leave=True):
        msg = choice.message.content
        text_sep = r'<sep>\n|<sep>'
        examples = re.split(text_sep, msg)
        print(df)
        for example in tqdm(examples, position=0, leave=True):
            if 'choice' in selected_field:
                print(df['id'].values.tolist()[idx])
                example = get_copa_prefix(df['question'].values.tolist()[idx], False)+ " " + example.lower()
            copa_like.append(example)
            idx+=1
    return copa_like

def create_winogrande_dataset_from_response(response,start_pos, df):
    choices = response.choices
    id = start_pos
    winogrande_like = []
    for choice in tqdm(choices, position=0, leave=True):
        msg = choice.message.content
        text_sep = r'<sep>\n'
        examples = re.split(text_sep, msg)
        for example in tqdm(examples, position=0, leave=True):
            answer = df['answer'][id]
            answer_text = df['option1'][id] if df['answer'][id]==1 else df['option2'][id] 
            translated_text = example.replace("Text: ", "").strip()
            translated_text = mask_answer(translated_text, answer_text)
            record = {
                "id": id,
                "sentence": translated_text,
                "original_sentence": df['sentence'][id],
                "option1": df['option1'][id],
                "option2": df['option2'][id],
                "label": df['answer'][id]
            }
            winogrande_like.append(record)
            id += 1
    return winogrande_like


def generate_winogrande_like_dataset(client, args):
    system_prompt = SYSTEM_INSTRUCTION
    batch_size, temperature = args.batch_size, args.temperature
    messages=[{"role": "system", "content": system_prompt}]
    instruction = "Please translate the following texts in social media English while preserving the semantic and also the.\n\n"
    dataset = load_dataset("winogrande", "winogrande_debiased", split="validation")
    df = dataset.to_pandas()
    num_rows = df.shape[0]
    num_samples = num_rows
    whole_outputs = []
    for start_pos in tqdm(range(0, num_rows, batch_size)):
        batch_df = df.iloc[start_pos:batch_size+start_pos]
        prompts = []
        for index, row in tqdm(batch_df.iterrows(), position=0, leave=True):
            sentence = row['sentence']
            answer = row['option1'] if row['answer']==1 else row['option2']
            complete_sentence = sentence.replace("_", answer)
            prompt = f'Text: {complete_sentence}'
            print(prompt)
            prompts.append(prompt)
        prompts = "<sep>\n".join(prompts)
        output_size = batch_size
        whole_instruction = instruction + prompts
        messages.append({"role": "user", "content": whole_instruction})
        message_length = num_tokens_from_string(whole_instruction)
        max_tokens = MAX_TOKEN_SIZE-message_length
        responses = client.chat.completions.create(
        model=LLM_TYPE,
        messages = messages,
        max_tokens = max_tokens
        )
        print(responses)
        batch_output = create_winogrande_dataset_from_response(responses, start_pos, df)
    whole_output = whole_output.extend(batch_output)
    with open("winogrande_eng-socmed_eng.json", "w") as f:
        json.dump(whole_output, f, indent=4)
        
def get_or_empty(lst, idx):
    if idx > len(lst)-1:
        return ""
    return lst[idx]

def create_siqa_dataset(translation, batch_df, start_pos):
    id = start_pos
    siqa_batch = []
    for index, row in tqdm(batch_df.iterrows(), position=0, leave=True):
        record = {
            "id": id,
            "context": row['context'],
            "translated_context": get_or_empty(translation['context'], id-start_pos),
            "question": row['question'],
            "translated_question": get_or_empty(translation['question'], id-start_pos),
            "answerA": row['answerA'],
            "translated_answerA": get_or_empty(translation['answerA'], id-start_pos),
            "answerB": row['answerB'],
            "translated_answerB": get_or_empty(translation['answerB'], id-start_pos),
            "answerC": row['answerC'],
            "translated_answerC": get_or_empty(translation['answerC'], id-start_pos),
            "label": row['label']
        } 
        siqa_batch.append(record)
        id +=1
    return siqa_batch

def create_copa_dataset(translation, batch_df, start_pos):
    id = start_pos
    copa_batch = []
    for index, row in tqdm(batch_df.iterrows(), position=0, leave=True):
        record = {
            "id": row['id'],
            "premise": row['premise'],
            "translated_premise": get_or_empty(translation['premise'], id-start_pos),
            "question": row['question'],
            "choice1": get_copa_prefix(row['question'], True) + " " + row['choice1'].lower(),
            "translated_choice1": get_or_empty(translation['choice1'], id-start_pos),
            "choice2": get_copa_prefix(row['question'], True) + " " + row['choice2'].lower(),
            "translated_choice2": get_or_empty(translation['choice2'], id-start_pos),
            "label": row['label']
        } 
        copa_batch.append(record)
        id +=1
    return copa_batch

def generate_siqa_like_dataset(client, args):
    system_prompt = SYSTEM_INSTRUCTION
    batch_size, temperature = args.batch_size, args.temperature
   
    instruction = "Please translate the following texts separated by <sep> to social media English and dont delete/ignore the <sep> on the translation result\n\n"
    dataset = load_dataset("social_i_qa", split="validation")
    df = dataset.to_pandas()
    num_rows = df.shape[0]
    num_samples = num_rows
    whole_outputs = []
    error_trials_limit = 10
    fields = ['context', 'question', 'answerA', 'answerB', 'answerC']
    for start_pos in tqdm(range(0, num_rows, batch_size)):
        batch_df = df.iloc[start_pos:batch_size+start_pos]
        
        translated_fields = dict()
        for field in tqdm(fields):
            prompts = []
            messages=[{"role": "system", "content": system_prompt}]
            for index, row in tqdm(batch_df.iterrows(), position=0, leave=True):
                prompt = row[field]
                prompts.append(prompt)
            prompts = "<sep>\n".join(prompts)
            output_size = batch_size
            whole_instruction = instruction + prompts
            messages.append({"role": "user", "content": whole_instruction})
            error_cnt = 0
            is_success = False
            while not is_success and error_cnt < error_trials_limit:
                try:
                    responses = client.chat.completions.create(
                    model=LLM_TYPE,
                    messages = messages
                    )
                    print(responses)
                    is_success = True
                except Exception as e:
                    print(e)
                    error_cnt += 1
            
            siqa_column = get_translated_siqa_fields_from_response(responses, batch_df)
            translated_fields[field] = siqa_column
        batch_output = create_siqa_dataset(translated_fields, batch_df, start_pos)
        whole_outputs.extend(batch_output)
    with open("siqa_eng-socmed_eng.json", "w") as f:
        json.dump(whole_outputs, f, indent=4)

def generate_copa_like_dataset(client, args):
    system_prompt = SYSTEM_INSTRUCTION
    batch_size, temperature = args.batch_size, args.temperature
   
    instruction = "Please translate the following texts separated by <sep> to social media English and dont delete/ignore the <sep> on the translation result\n\n"
    dataset = load_dataset("pkavumba/balanced-copa", split="test")
    df = dataset.to_pandas()
    num_rows = df.shape[0]
    num_samples = num_rows
    whole_outputs = []
    error_trials_limit = 10
    fields = ['premise', 'choice1', 'choice2']
    for start_pos in tqdm(range(0, num_rows, batch_size)):
        batch_df = df.iloc[start_pos:batch_size+start_pos]
        
        translated_fields = dict()
        for field in tqdm(fields):
            prompts = []
            messages=[{"role": "system", "content": system_prompt}]
            for index, row in tqdm(batch_df.iterrows(), position=0, leave=True):
                prompt = row[field]
                prompts.append(prompt)
            prompts = "<sep>\n".join(prompts)
            output_size = batch_size
            whole_instruction = instruction + prompts
            messages.append({"role": "user", "content": whole_instruction})
            error_cnt = 0
            is_success = False
            while not is_success and error_cnt < error_trials_limit:
                try:
                    responses = client.chat.completions.create(
                    model=LLM_TYPE,
                    messages = messages
                    )
                    print(responses)
                    is_success = True
                except Exception as e:
                    print(e)
                    error_cnt += 1
            
            copa_column = get_translated_copa_fields_from_response(responses, field, batch_df)
            translated_fields[field] = copa_column
        batch_output = create_copa_dataset(translated_fields, batch_df, start_pos)
        whole_outputs.extend(batch_output)
    with open("copa_eng-socmed_eng.json", "w") as f:
        json.dump(whole_outputs, f, indent=4)

if __name__ == "__main__":
    client = OpenAI()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="winogrande, siqa, copa")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.825)

    args = parser.parse_args() 
    dataset_name = args.dataset_name
    if dataset_name == 'winogrande':
        generate_winogrande_like_dataset(client, args)
    elif dataset_name == 'siqa':
        generate_siqa_like_dataset(client, args)
    elif dataset_name == 'copa':
        generate_copa_like_dataset(client, args)