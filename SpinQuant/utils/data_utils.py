# coding=utf-8

import random
from typing import Any, Dict, List
import datasets
import torch
import transformers


def _get_tokenizer(model: str, tokenizer=None):
    return tokenizer or transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

def get_wikitext2(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False):
    tokenizer = _get_tokenizer(model, tokenizer)
    split = "test" if eval_mode else "train"
    data = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[split]
    text = "\n\n".join(data["text"])
    encodings = tokenizer(text, return_tensors="pt")
    
    if eval_mode:
        return encodings
    
    random.seed(seed)
    length = encodings.input_ids.shape[1]
    trainloader = []
    
    for _ in range(nsamples):
        start = random.randint(0, length - seqlen - 1)
        end = start + seqlen
        inp = encodings.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    return trainloader

def get_json_data(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, json_data_path="", eval_mode=False):
    tokenizer = _get_tokenizer(model, tokenizer)
    calibration_datasets = datasets.load_dataset('json', data_files=json_data_path, split='train[:]')

    cols = calibration_datasets.column_names
    if "input" in cols and "output" in cols:
        text = "\n\n".join(calibration_datasets["input"] + calibration_datasets["output"])
    elif "text" in cols:
        text = "\n\n".join(calibration_datasets["text"])
    else:
        raise ValueError(f"Unsupported calibration columns: {cols}")
    encodings = tokenizer(text, return_tensors="pt")
    
    if eval_mode:
        return encodings
    
    random.seed(seed)
    trainloader = []
    
    for _ in range(nsamples):
        start = random.randint(0, encodings.input_ids.shape[1] - seqlen - 1)
        end = start + seqlen
        inp = encodings.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    return trainloader

class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size: int = 1024) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = [self.tokenize_function(d) for d in dataset]
        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        if "input" in examples and "output" in examples:
            return self.tokenizer(examples["input"] + examples["output"])
        elif "text" in examples:
            return self.tokenizer(examples["text"])
        else:
            raise KeyError("Dataset must contain 'input' and 'output' keys, or 'text' key.")

    def group_texts(self, examples):
        concatenated_examples = {}
        
        for d in examples:
            for key in d.keys():
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                concatenated_examples[key].extend(d[key])
        
        total_length = len(concatenated_examples["input_ids"])
        
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        
        return result
