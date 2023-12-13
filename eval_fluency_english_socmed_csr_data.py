from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
import numpy as np
from evaluate import load
import pandas as pd
import argparse
from tqdm import tqdm
import transformers
import json
import random
from sklearn.metrics import f1_score, classification_report
import glob
import os


def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids
    preds = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, preds, average='weighted')
    return {"f1": f1}

def tokenize_function(row, text_column, tokenizer, max_length):
    return tokenizer(row[text_column], padding="max_length", truncation=True, max_length=max_length)

def preprocess_dataset(text_column, dataset, tokenizer, max_length):
    print("Preprocessing...")
    dataset = dataset.map(lambda row: tokenize_function(row, text_column, tokenizer, max_length))
    print("Preprocessing finished")
    return dataset

def load_input_dataset(input_file, text_column, tokenizer, max_length):
    assert input_file.endswith('.json')
    input_df = pd.read_json(input_file)
    input_dataset = preprocess_dataset(text_column, input_df, tokenizer, max_length)
    return input_dataset

def load_model_and_tokenizer(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def preprocess_columns(socmed_dataset, nonsocmed_dataset):
    socmed_dataset = socmed_dataset.select_columns(['content', 'label'])
    nonsocmed_dataset = nonsocmed_dataset.select_columns(['summary', 'label'])

    socmed_dataset = socmed_dataset.rename_column("content", "text")
    nonsocmed_dataset = nonsocmed_dataset.rename_column("summary", "text")

    return socmed_dataset, nonsocmed_dataset


def sample_dataset(socmed_dataset, nonsocmed_dataset, seed):
    print("Sampling dataset...")
    socmed_len = len(socmed_dataset)
    nonsocmed_len = len(nonsocmed_dataset)
    print(nonsocmed_len)
    random.seed(seed)
    if nonsocmed_len < socmed_len:
        socmed_idxs = random.sample(list(np.arange(0, socmed_len)), nonsocmed_len)
        socmed_dataset = socmed_dataset.select(socmed_idxs)
    else:
        nonsocmed_idxs = random.sample(list(np.arange(0, nonsocmed_len)), socmed_len)
        nonsocmed_dataset = nonsocmed_dataset.select(nonsocmed_idxs)
    print("Sampling finished")
    return socmed_dataset, nonsocmed_dataset

def merge_datasets(socmed_dataset, nonsocmed_dataset, seed):
    merged_datasets = concatenate_datasets([socmed_dataset, nonsocmed_dataset])
    merged_datasets = merged_datasets.shuffle(seed=seed)

    split_data = merged_datasets.train_test_split(test_size=0.2)
    train_data = split_data["train"]
    val_data = split_data["test"]
    train_data.to_json("train-socmed-classification.json")
    val_data.to_json("val-socmed-classification.json")
    return train_data, val_data

def add_labels(socmed_dataset, nonsocmed_dataset):
    socmed_labels = [1]
    nonsocmed_labels = [0]
    
    socmed_len = len(socmed_dataset)
    nonsocmed_len = len(nonsocmed_dataset)

    socmed_labels = socmed_labels*socmed_len
    nonsocmed_labels = nonsocmed_labels*nonsocmed_len
    print(nonsocmed_dataset)
    socmed_dataset = socmed_dataset.add_column('label', socmed_labels)
    nonsocmed_dataset = nonsocmed_dataset.add_column('label', nonsocmed_labels)

    return socmed_dataset, nonsocmed_dataset

def load_train_dataset(seed):
    socmed_dataset = load_dataset("webis/tldr-17", split='train')
    nonsocmed_dataset = load_dataset("yanbozhang/wikipedia-summary-only", split='train')
    
    socmed_dataset, nonsocmed_dataset = sample_dataset(socmed_dataset, nonsocmed_dataset, seed)

    socmed_dataset, nonsocmed_dataset = add_labels(socmed_dataset, nonsocmed_dataset)

    socmed_dataset, nonsocmed_dataset = preprocess_columns(socmed_dataset, nonsocmed_dataset)

    train_dataset, val_dataset = merge_datasets(socmed_dataset, nonsocmed_dataset, seed)

    return train_dataset, val_dataset


def init_trainer(model, tokenizer, train_dataset, val_dataset, args):
    training_args = TrainingArguments(
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.val_batch_size,
    logging_dir=args.logging_dir,
    logging_steps=args.logging_steps,
    evaluation_strategy=args.evaluation_strategy,
    save_strategy=args.save_strategy,
    output_dir=args.checkpoints_dir,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    push_to_hub=False,
    logging_first_step=True,
    metric_for_best_model=args.metric_for_best_model,
    greater_is_better=args.greater_is_better,
    load_best_model_at_end=args.load_best_model_at_end
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer

def evaluate_fluency_with_model(trainer, tokenizer, text_columns, test_file, max_length):
    for text_col in text_columns:
        for text_col_type in text_col:
            print(f"Text column: {text_col_type}")
            test_dataset = load_input_dataset(test_file, text_col_type, tokenizer, max_length)
            results = trainer.predict(test_dataset)
            results = np.argmax(results.predictions, axis=-1)
            pred_labels = []
            truth_labels = []
            with open(test_file, "r") as file:
                lst = json.load(file)
                for instance, pred in zip(lst, results):
                    pred_labels.append(int(pred))
                    truth_labels.append(int(instance['label']))
            print(classification_report(truth_labels, pred_labels))

def load_dataset_from_file(train_file, val_file):
    train_dataset = load_dataset('json', data_files=train_file, split='train')
    val_dataset = load_dataset('json', data_files=val_file, split='train')
    return train_dataset, val_dataset

def get_best_checkpoint(model_name, checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        raise ValueError(
            f"Output directory ({checkpoints_dir}) does not exist. Please train the model first."
        )

    # Find the best model checkpoint
    ckpt_paths = sorted(
        glob.glob(os.path.join(checkpoints_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1]),
    )

    if not ckpt_paths:
        raise ValueError(
            f"Output directory ({checkpoints_dir}) does not contain any checkpoint. Please train the model first."
        )

    state = TrainerState.load_from_json(
        os.path.join(ckpt_paths[-1], "trainer_state.json")
    )
    best_model_path = state.best_model_checkpoint or model_name
    if state.best_model_checkpoint is None:
        print("No best model checkpoint found. Using the default model checkpoint.")
    print(f"Best model path: {best_model_path}")
    return best_model_path

def evaluate_fluency_using_evaluation(test_file, text_columns):
    with open(test_file, "r") as file:
        lst = json.load(file)
    for text_col in text_columns:
        print(f"Text column: {text_col}")
        translated = []
        og = []
        for instance in tqdm(lst):
            translated.append(instance[text_col[1]])
            og.append(instance[text_col[0]])
        bertscore = load("bertscore")
        bleu = load("bleu")
        bleu_res = bleu.compute(predictions=translated, references=og)
        print(f"BLEU:\n{bleu_res}")
        bert_res = bertscore.compute(predictions=translated, references=og, lang="en", model_type="microsoft/deberta-xlarge-mnli")
        f1_bert_res = sum(bert_res['f1'])/len(bert_res['f1'])
        p_bert_res = sum(bert_res['precision'])/len(bert_res['precision'])
        r_bert_res = sum(bert_res['recall'])/len(bert_res['recall'])
        print(f"BERT SCORE:\nF1: {f1_bert_res}\nP: {p_bert_res}\nR: {r_bert_res}")

def main(args):
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    transformers.set_seed(args.seed)
    text_columns = eval(args.pair_fields)
    if args.do_train or args.do_predict:
        if args.classifier_train_data is None or args.classifier_val_data is None:
            train_data, val_data = load_train_dataset(args.seed)
        else:
            train_data, val_data = load_dataset_from_file(args.classifier_train_data, args.classifier_val_data)
        train_data, val_data = preprocess_dataset('text', train_data, tokenizer, args.max_length), preprocess_dataset('text', val_data, tokenizer, args.max_length)
        if args.do_train:
            print("Training...")
            print("*** Train Dataset ***")
            print(f"Number of samples: {len(train_data)}")
            print("*** Dev Dataset ***")
            print(f"Number of samples: {len(val_data)}")
            trainer = init_trainer(model, tokenizer, train_data, val_data, args)
            trainer.train()
            print("Training completed!")
        else:
            best_checkpoint = get_best_checkpoint(args.model_name, args.checkpoints_dir)
            model, tokenizer = load_model_and_tokenizer(best_checkpoint)
            trainer = init_trainer(model, tokenizer, train_data, val_data, args)

        evaluate_fluency_with_model(trainer, tokenizer, text_columns, args.test_file, args.max_length)
    evaluate_fluency_using_evaluation(args.test_file, text_columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_file', type=str)

    parser.add_argument('--classifier_train_data', type=str, default=None)
    parser.add_argument('--classifier_val_data', type=str, default=None)

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')

    parser.add_argument('--checkpoints_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--logging_dir', type=str, default='./logs')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_train_epochs', type=int, default=30)
    parser.add_argument('--model_name', type=str, default="cardiffnlp/twitter-roberta-base")
    parser.add_argument('--metric_for_best_model', type=str, default="eval_f1") 
    parser.add_argument('--greater_is_better', type=bool, default=True) 
    parser.add_argument('--evaluation_strategy', type=str, default="epoch") 
    parser.add_argument('--save_strategy', type=str, default="epoch")
    parser.add_argument('--load_best_model_at_end', type=bool, default=True) 

    parser.add_argument('--pair_fields', type=str)
    args = parser.parse_args()
    
    main(args)

