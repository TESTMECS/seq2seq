"""
Data loading and preprocessing for machine translation task.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Tuple, Dict, Any
from datasets import load_dataset
from tokenizers import Tokenizer


class TranslationDataset(Dataset):
    """Dataset for machine translation task."""

    def __init__(
        self,
        sources: List[List[int]],
        targets: List[List[int]],
        source_lang: str = "en",
        target_lang: str = "fr",
    ):
        """Initialize dataset with tokenized source and target sentences."""
        self.sources = sources
        self.targets = targets
        self.source_lang = source_lang
        self.target_lang = target_lang

        assert len(sources) == len(targets), "Source and target lists must have the same length"

    def __len__(self) -> int:
        """Return the number of sentence pairs."""
        return len(self.sources)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        """Return a single source-target pair."""
        return self.sources[idx], self.targets[idx]


class BatchSampler:
    """
    Custom batch sampler that groups sentences with similar lengths.
    This helps to minimize padding and speeds up training.
    """

    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True):
        """Initialize the sampler with sequence lengths."""
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        """Yield batches of indices with similar lengths."""
        # Sort indices by length
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            np.random.shuffle(indices)

        # Sort indices by length (to group similar lengths)
        indices.sort(key=lambda i: self.lengths[i])

        # Group indices into batches of similar length
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batches.append(indices[i : i + self.batch_size])

        # Shuffle the batches if needed
        if self.shuffle:
            np.random.shuffle(batches)

        # Yield each batch
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches."""
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def collate_fn(batch: List[Tuple[List[int], List[int]]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to pad sequences in a batch.
    """
    # Separate sources and targets
    sources, targets = zip(*batch)

    # Filter out any None or empty values
    valid_pairs = [(s, t) for s, t in zip(sources, targets) if s and t]
    if len(valid_pairs) != len(batch):
        print(f"Warning: Found {len(batch) - len(valid_pairs)} invalid pairs in batch")

    if not valid_pairs:
        # Return empty tensors if no valid pairs
        return {
            "src": torch.zeros((0, 0), dtype=torch.long),
            "tgt": torch.zeros((0, 0), dtype=torch.long),
            "src_mask": torch.zeros((0, 0), dtype=torch.float),
            "tgt_mask": torch.zeros((0, 0), dtype=torch.float),
            "src_lengths": torch.zeros(0, dtype=torch.long),
            "tgt_lengths": torch.zeros(0, dtype=torch.long),
        }

    # Use valid pairs only
    sources, targets = zip(*valid_pairs)

    # Convert to tensors and pad
    src_tensors = [torch.tensor(s, dtype=torch.long) for s in sources]
    tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in targets]

    # Padding
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=0)

    # Create source and target masks (1 for non-pad, 0 for pad)
    src_mask = (src_padded != 0).float()
    tgt_mask = (tgt_padded != 0).float()

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "src_lengths": torch.tensor([len(s) for s in sources], dtype=torch.long),
        "tgt_lengths": torch.tensor([len(t) for t in targets], dtype=torch.long),
    }


def load_tokenizers(src_path: str, tgt_path: str) -> Tuple[Tokenizer, Tokenizer]:
    """Load tokenizers from JSON files."""
    src_tokenizer = Tokenizer.from_file(src_path)
    tgt_tokenizer = Tokenizer.from_file(tgt_path)
    return src_tokenizer, tgt_tokenizer


def load_iwslt_data(
    src_lang: str = "en", tgt_lang: str = "fr", split: str = "train"
) -> List[Tuple[str, str]]:
    """Load IWSLT dataset for the specified languages and split."""
    # The correct format is 'iwslt2017-src-tgt' for the config name
    config_name = f"iwslt2017-{src_lang}-{tgt_lang}"
    dataset = load_dataset("iwslt2017", config_name, split=split)

    # Extract sentence pairs
    pairs = []
    for item in dataset:
        src_text = item["translation"][src_lang]
        tgt_text = item["translation"][tgt_lang]
        pairs.append((src_text, tgt_text))

    return pairs


def prepare_data(
    src_lang: str = "en",
    tgt_lang: str = "fr",
    batch_size: int = 32,
    max_length: int = 100,
) -> Dict[str, Any]:
    """
    Prepare data for training and evaluation.

    Returns:
        Dictionary containing DataLoaders and tokenizers.
    """
    # Load and tokenize data
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        f"{src_lang}_tokenizer.json", f"{tgt_lang}_tokenizer.json"
    )

    # Load datasets
    train_pairs = load_iwslt_data(src_lang, tgt_lang, "train")
    val_pairs = load_iwslt_data(src_lang, tgt_lang, "validation")
    test_pairs = load_iwslt_data(src_lang, tgt_lang, "test")

    # Tokenize datasets
    def tokenize_pairs(pairs):
        src_tokenized = []
        tgt_tokenized = []

        for src_text, tgt_text in pairs:
            # Skip empty or None texts
            if not src_text or not tgt_text:
                continue

            try:
                src_tokens = src_tokenizer.encode(src_text).ids
                tgt_tokens = tgt_tokenizer.encode(tgt_text).ids

                # Skip if tokenization failed
                if not src_tokens or not tgt_tokens:
                    continue

                # Truncate if needed
                if len(src_tokens) > max_length:
                    src_tokens = src_tokens[:max_length]
                if len(tgt_tokens) > max_length:
                    tgt_tokens = tgt_tokens[:max_length]

                src_tokenized.append(src_tokens)
                tgt_tokenized.append(
                    [tgt_tokenizer.token_to_id("[BOS]")]
                    + tgt_tokens
                    + [tgt_tokenizer.token_to_id("[EOS]")]
                )
            except Exception as e:
                print(f"Error tokenizing pair: {e}")
                continue

        return src_tokenized, tgt_tokenized

    # Tokenize all datasets
    train_src, train_tgt = tokenize_pairs(train_pairs)
    val_src, val_tgt = tokenize_pairs(val_pairs)
    test_src, test_tgt = tokenize_pairs(test_pairs)

    # Create datasets
    train_dataset = TranslationDataset(train_src, train_tgt, src_lang, tgt_lang)
    val_dataset = TranslationDataset(val_src, val_tgt, src_lang, tgt_lang)
    test_dataset = TranslationDataset(test_src, test_tgt, src_lang, tgt_lang)

    # Create batch samplers
    # Make sure no empty sequences cause problems in samplers
    train_src_lengths = [len(x) if x else 1 for x in train_src]
    val_src_lengths = [len(x) if x else 1 for x in val_src]
    test_src_lengths = [len(x) if x else 1 for x in test_src]

    train_sampler = BatchSampler(train_src_lengths, batch_size, shuffle=True)
    val_sampler = BatchSampler(val_src_lengths, batch_size, shuffle=False)
    test_sampler = BatchSampler(test_src_lengths, batch_size, shuffle=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "src_tokenizer": src_tokenizer,
        "tgt_tokenizer": tgt_tokenizer,
        "src_vocab_size": src_tokenizer.get_vocab_size(),
        "tgt_vocab_size": tgt_tokenizer.get_vocab_size(),
    }
