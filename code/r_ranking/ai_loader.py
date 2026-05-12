
import os
import random
import time
from dataclasses import dataclass, field

import torch
from transformers import DataCollatorWithPadding


@dataclass
class AiCollator(DataCollatorWithPadding):
    """
    data collector for LLM Detect AI Generated Text task
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        labels = None
        if "generated" in features[0].keys():
            labels = [feature["generated"] for feature in features]

        features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            } for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        if labels is not None:
            batch["labels"] = labels

        tensor_keys = [
            "input_ids",
            "attention_mask",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        return batch


@dataclass
class AiCollatorTrain(DataCollatorWithPadding):
    """
    data collector for LLM Detect AI Generated Text task
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        [setattr(self, k, v) for k, v in self.kwargs.items()]

        # Debug: print available columns
        print("=="*40)
        print(f"Available columns in train_ds: {self.train_ds.column_names}")
        print(f"First 3 prompt_ids: {self.train_ds['prompt_id'][:3]}")
        print("=="*40)

        # mappings
        example2idx = dict()
        example_ids = list(self.train_ds["id"])

        for idx in range(len(example_ids)):
            example2idx[example_ids[idx]] = idx
        self.example2idx = example2idx

        # Build prompt_id mappings - use direct column access
        prompt2ids = dict()
        all_prompt_ids = list(self.train_ds["prompt_id"])
        for idx, eid in enumerate(example_ids):
            prompt_id = all_prompt_ids[idx]
            if prompt_id not in prompt2ids:
                prompt2ids[prompt_id] = []
            prompt2ids[prompt_id].append(eid)
        self.prompt2ids = prompt2ids
        self.prompt_ids = list(prompt2ids.keys())

        seed = int(time.time() * 1000) + os.getpid()
        self.rng = random.Random(seed)

        print("=="*40)
        print(f"setting random seed in data collator as: {seed}")
        print(f"Total unique prompt_ids: {len(self.prompt_ids)}")
        print(f"Prompt distribution (first 10): {[(p, len(ids)) for p, ids in list(self.prompt2ids.items())[:10]]}")
        print("=="*40)

    def process_features(self, example_ids):
        updated_features = []
        for eid in example_ids:
            example = dict()

            example["id"] = eid
            ex_info = self.train_ds[self.example2idx[eid]]

            # use fields
            example["input_ids"] = ex_info["input_ids"]
            example["attention_mask"] = ex_info["attention_mask"]
            example["generated"] = ex_info["generated"]
            updated_features.append(example)

        return updated_features

    def __call__(self, features):
        bs = len(features)

        if self.rng.random() < 0.8:
            # Try to get examples from a single prompt_id
            selected_prompt_id = self.rng.choice(self.prompt_ids)
            prompt_examples = self.prompt2ids[selected_prompt_id]
            
            # If we have enough examples in this prompt, use them
            if len(prompt_examples) >= bs:
                selected_example_ids = self.rng.sample(prompt_examples, k=bs)
                features = self.process_features(selected_example_ids)
            else:
                # Not enough examples - create batch from multiple prompts
                # Collect examples from multiple prompt_ids to fill the batch
                selected_example_ids = []
                available_prompts = [p for p in self.prompt_ids if p != selected_prompt_id]
                
                # Add all examples from the selected prompt
                selected_example_ids.extend(prompt_examples)
                
                # Add examples from other prompts until we fill the batch
                self.rng.shuffle(available_prompts)
                for prompt_id in available_prompts:
                    if len(selected_example_ids) >= bs:
                        break
                    prompt_examples = self.prompt2ids[prompt_id]
                    # Add one random example from this prompt
                    selected_example_ids.append(self.rng.choice(prompt_examples))
                
                # Trim to exact batch size
                selected_example_ids = selected_example_ids[:bs]
                features = self.process_features(selected_example_ids)

        labels = None
        if "generated" in features[0].keys():
            labels = [feature["generated"] for feature in features]

        features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            } for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        if labels is not None:
            batch["labels"] = labels

        tensor_keys = [
            "input_ids",
            "attention_mask",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        return batch

# ---


def show_batch(batch, tokenizer, n_examples=16, task='training', print_fn=print):
    print_fn("##"*40)
    bs = batch['input_ids'].size(0)
    print_fn(f"batch size: {bs}")

    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")

    n_examples = min(n_examples, bs)
    print_fn(f"Showing {n_examples} from a {task} batch...")

    print_fn("\n\n")
    for idx in range(n_examples):
        print_fn(f"Example {idx+1}")
        print_fn(f"Input:\n\n{tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)}")
        # print("\n\n")

        if "infer" not in task.lower():
            print_fn("--"*20)
            labels = batch['labels'][idx]
            print_fn(f"Label: {labels}")
        print_fn('=='*40)
    print_fn("##"*40)
