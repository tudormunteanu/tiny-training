import re
from typing import Dict

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler


def setup_model():
    # Using a smaller CodeT5 model suitable for the free tier
    model_name = "Salesforce/codet5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def prepare_dataset():
    # Load Python subset of CodeSearchNet
    dataset = load_dataset(
        "code_search_net", "python", split="train[:1000]", trust_remote_code=True
    )  # Limited to 1000 examples for free tier

    def extract_function_info(example: Dict) -> Dict:
        """Extract clean function definitions and docstrings."""
        code = example["whole_func_string"]

        # Basic filtering for API-style functions
        if not code.strip().startswith("def "):
            # Empty strings are better handled downstream.
            return {
                "function": "",
                "documentation": "",
                "input": "",
                "output": ""
            }

        # Remove multiple newlines and standardize spacing
        code = re.sub(r"\n\s*\n", "\n", code)
        docstring = example["func_documentation_string"].strip()

        return {
            "function": code,
            "documentation": docstring,
            "input": f"Write a Python function that: {docstring}",
            "output": code,
        }

    # Process and filter the dataset
    processed_dataset = dataset.map(extract_function_info)
    # Filter out empty entries after mapping
    processed_dataset = processed_dataset.filter(lambda x: x["function"] != "")

    return processed_dataset


def tokenize_data(examples, tokenizer, max_length=512):
    """Tokenize inputs and outputs for training."""
    # Batch tokenization for inputs
    model_inputs = tokenizer(
        examples['input'],
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    
    # Batch tokenization for outputs
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['output'],
            max_length=max_length,
            padding='max_length',
            truncation=True
        ).input_ids
    
    model_inputs['labels'] = labels
    return model_inputs


def train():
    model, tokenizer = setup_model()
    dataset = prepare_dataset()

    # Training configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Hyperparameters
    batch_size = 8
    num_epochs = 3
    learning_rate = 5e-5
    max_length = 512

    # Modify the dataset mapping
    tokenized_dataset = dataset.map(
        lambda x: tokenize_data(x, tokenizer, max_length),
        batched=True,
        batch_size=16,  # Explicit batch size for processing
        remove_columns=dataset.column_names,
    )

    def collate_fn(examples):
        return {
            'input_ids': torch.stack([torch.tensor(example['input_ids']) for example in examples]).to(device),
            'attention_mask': torch.stack([torch.tensor(example['attention_mask']) for example in examples]).to(device),
            'labels': torch.stack([torch.tensor(example['labels']) for example in examples]).to(device)
        }

    train_dataloader = DataLoader(
        tokenized_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Training loop
    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description(f"Loss: {loss.item():.4f}")

        # Save checkpoint after each epoch
        model.save_pretrained(f"checkpoint-epoch-{epoch}")
        tokenizer.save_pretrained(f"checkpoint-epoch-{epoch}")

    print("Training completed!")


if __name__ == "__main__":
    train()
