# coding=utf-8

"""  utility method to evaluate perplexity score on WikiText """
from itertools import chain
import torch
import jsonlines
from torch.utils.data import DataLoader, Dataset
from datasets import IterableDataset, load_dataset
from transformers import default_data_collator


class CustomDataset(Dataset):
    """
    Dataset for GPTQ-preprocessed tokens
    """
    def __init__(self, tokens, block_size=2048):
        self.full_tokens = tokens
        self.block_size = block_size
        self._len = len(tokens["input_ids"][0]) // block_size

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = (idx+1) * self.block_size

        input_ids = self.full_tokens["input_ids"][0, start_idx:end_idx]
        labels = input_ids.clone()
        attn_mask = self.full_tokens["attention_mask"][0, start_idx:end_idx]
        output = {"input_ids": input_ids,
                  "attention_mask": attn_mask,
                  "labels": labels}
        return output


class PreprocessGptqSplit:
    def __init__(self, tokenizer, block_size, add_special_tokens=True):
        self._block_size = block_size
        self._add_special_tokens = add_special_tokens
        self._tokenizer = tokenizer

    def preprocess(self, dataset):
        txt = [item['instruction'] + item['input'] + item['output'] for item in dataset]
        tokens = self._tokenizer("\n\n".join(txt), return_tensors="pt", add_special_tokens=self._add_special_tokens)
        dataset = CustomDataset(tokens, self._block_size)
        return dataset

class PreprocessSplit:
    def __init__(self, tokenizer, block_size, add_special_tokens=True):
        self._block_size = block_size
        self._add_special_tokens = add_special_tokens
        self._tokenizer = tokenizer

    def _tokenize_fn(self, examples):
        return self._tokenizer(examples['input'], return_token_type_ids=False, add_special_tokens=self._add_special_tokens)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def _group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: torch.tensor(list(chain(*examples[k]))) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self._block_size:
            total_length = (total_length // self._block_size) * self._block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self._block_size] for i in range(0, total_length, self._block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def preprocess(self, dataset):
        map_kwargs = {
            "num_proc": None,
            "load_from_cache_file": True,
            "desc": "Running tokenizer on dataset",
        }
        func = lambda item: {
            'instruction' : "",
            'input' : item['instruction'] + item['input'] + item['output'],
        }
        text_dataset = dataset.map(func)
        tokenized_dataset = text_dataset.map(
            self._tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
            **(map_kwargs if not isinstance(text_dataset, IterableDataset) else {}),
        )

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        print(f"[Load Dataset] grouped train dataset")
        map_kwargs["desc"] = f"Grouping texts in chunks of {self._block_size}"
        dataset = tokenized_dataset.map(
            self._group_texts,
            batched=True,
            **(map_kwargs if not isinstance(tokenized_dataset, IterableDataset) else {}),
        )

        return dataset

def get_qwen2_dataset(tokenizer, block_size, train_file, train_batch_size = 1, test_batch_size = 1):
    dataset = {}
    dataset['train'] = load_dataset('json', data_files=train_file, split='train[:-300]')
    dataset['train'] = PreprocessSplit(tokenizer, block_size).preprocess(dataset['train'])
    dataset['test'] = load_dataset('json', data_files=train_file, split='train[-300:]')
    dataset['test'] = PreprocessGptqSplit(tokenizer, block_size).preprocess(dataset['test'])
    train_dataloader = DataLoader(dataset['train'], shuffle=False, batch_size=train_batch_size, collate_fn=default_data_collator)
    test_dataloader = DataLoader(dataset['test'], shuffle=False, batch_size=test_batch_size, collate_fn=default_data_collator)

    return train_dataloader, test_dataloader, dataset
