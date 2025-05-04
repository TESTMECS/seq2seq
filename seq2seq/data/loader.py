"""
Data loading utilities for machine translation.
(a)
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional
from datasets import load_dataset
from tokenizers import Tokenizer

from seq2seq.data.dataset import TranslationDataset
from seq2seq.data.sampler import BatchSampler
from seq2seq.data.collate import collate_fn


def load_tokenizers(src_lang: str, tgt_lang: str) -> Tuple[Tokenizer, Tokenizer]:
    """
    Load tokenizers for source and target languages.

    Args:
        src_lang: Source language code
        tgt_lang: Target language code

    Returns:
        Tuple of source and target tokenizers
    """
    src_tokenizer = Tokenizer.from_file(f"{src_lang}_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file(f"{tgt_lang}_tokenizer.json")
    return src_tokenizer, tgt_tokenizer


def load_iwslt_data(src_lang: str, tgt_lang: str, split: str) -> Tuple[List[str], List[str]]:
    """
    Load IWSLT dataset for specified languages and split.

    Args:
        src_lang: Source language code
        tgt_lang: Target language code
        split: Dataset split (train, validation, test)

    Returns:
        Tuple of source and target texts
    """
    dataset = load_dataset("iwslt2017", f"iwslt2017-{src_lang}-{tgt_lang}", split=split)
    src_texts = [item["translation"][src_lang] for item in dataset]
    tgt_texts = [item["translation"][tgt_lang] for item in dataset]
    return src_texts, tgt_texts


def prepare_data(
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_length: int,
    data_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare data for training and evaluation.

    Args:
        src_lang: Source language code
        tgt_lang: Target language code
        batch_size: Batch size
        max_length: Maximum sequence length
        data_cache_dir: Directory to cache data

    Returns:
        Dictionary containing dataloaders and tokenizers
    """
    # Load tokenizers
    src_tokenizer, tgt_tokenizer = load_tokenizers(src_lang, tgt_lang)

    # Load data
    train_src, train_tgt = load_iwslt_data(src_lang, tgt_lang, "train")
    val_src, val_tgt = load_iwslt_data(src_lang, tgt_lang, "validation")
    test_src, test_tgt = load_iwslt_data(src_lang, tgt_lang, "test")

    # Tokenize data
    def tokenize_pairs(src_texts, tgt_texts, max_length):
        src_tokenized = []
        tgt_tokenized = []
        for src, tgt in zip(src_texts, tgt_texts):
            src_tokens = src_tokenizer.encode(src).ids
            tgt_tokens = tgt_tokenizer.encode(tgt).ids

            # Filter out sequences that are too long
            if len(src_tokens) <= max_length and len(tgt_tokens) <= max_length:
                src_tokenized.append(src_tokens)
                tgt_tokenized.append(tgt_tokens)
        return src_tokenized, tgt_tokenized

    train_src_tokenized, train_tgt_tokenized = tokenize_pairs(train_src, train_tgt, max_length)
    val_src_tokenized, val_tgt_tokenized = tokenize_pairs(val_src, val_tgt, max_length)
    test_src_tokenized, test_tgt_tokenized = tokenize_pairs(test_src, test_tgt, max_length)

    # Create datasets
    train_dataset = TranslationDataset(train_src_tokenized, train_tgt_tokenized)
    val_dataset = TranslationDataset(val_src_tokenized, val_tgt_tokenized)
    test_dataset = TranslationDataset(test_src_tokenized, test_tgt_tokenized)

    # Create batch samplers
    train_sampler = BatchSampler(
        dataset_length=len(train_dataset),
        lens=[len(x) for x in train_src_tokenized],
        batch_size=batch_size,
        shuffle=True,
    )
    val_sampler = BatchSampler(
        dataset_length=len(val_dataset),
        lens=[len(x) for x in val_src_tokenized],
        batch_size=batch_size,
        shuffle=False,
    )
    test_sampler = BatchSampler(
        dataset_length=len(test_dataset),
        lens=[len(x) for x in test_src_tokenized],
        batch_size=batch_size,
        shuffle=False,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
    )

    # Gather data info
    data_info = {
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
        "src_tokenizer": src_tokenizer,
        "tgt_tokenizer": tgt_tokenizer,
        "src_vocab_size": src_tokenizer.get_vocab_size(),
        "tgt_vocab_size": tgt_tokenizer.get_vocab_size(),
        "pad_idx": src_tokenizer.token_to_id("[PAD]"),
        "bos_idx": src_tokenizer.token_to_id("[BOS]"),
        "eos_idx": src_tokenizer.token_to_id("[EOS]"),
    }

    return data_info
