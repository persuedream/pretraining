import os
import pickle
import time

import torch
from filelock import FileLock
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer, logger
from transformers.tokenization_utils_base import PaddingStrategy


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)

        logger.info("Creating features from dataset file at %s", file_path)
        logger.info(f"block_size: {block_size}")
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                with open(file_path, encoding="utf-8") as f:
                    lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

                line_length = len(lines)
                i = 0
                delta = 10000
                self.examples = []
                logger.info(f"data set length: {len(lines)}, tokenizing...")
                while i + delta < line_length:
                    # if i > 80000:
                    #     break
                    logger.info(f"tokenizing line: {i} ~ {i+delta}..")
                    batch_encoding = tokenizer(lines[i:i+delta], add_special_tokens=True,
                                               padding="max_length", truncation=True, max_length=block_size)
                    if batch_encoding["input_ids"]:
                        logger.info(f"first input ids: {len(batch_encoding['input_ids'][0])}")
                    # assert all(len(x) == block_size for x in batch_encoding["input_ids"]), "some sample length are invalid!!"
                    self.examples.extend(batch_encoding["input_ids"])
                    i += delta
                if i < line_length <= i + delta:
                    logger.info(f"tokenizing line: {i} ~ {line_length}..")
                    batch_encoding = tokenizer(lines[i:], add_special_tokens=True,
                                               padding="max_length", truncation=True, max_length=block_size)
                    if batch_encoding["input_ids"]:
                        logger.info(f"first input ids: {len(batch_encoding['input_ids'][0])}")
                    # assert all(len(x) == block_size for x in batch_encoding["input_ids"]), "some sample length are invalid!!"
                    self.examples.extend(batch_encoding["input_ids"])

                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class CustomIterableDataset(IterableDataset):

    def __init__(self, file_path, tokenizer, block_size):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.length = None

        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        logger.info(f"block_size: {block_size}")

    def preprocess(self, text):
        batch_encoding = self.tokenizer(text.strip("\n"), add_special_tokens=True,
                                        padding="max_length", truncation=True, max_length=self.block_size)

        return torch.tensor(batch_encoding["input_ids"])

    def line_mapper(self, line):
        return self.preprocess(line)

    def __iter__(self):
        file_itr = open(self.file_path, encoding="utf-8")
        file_itr = filter(lambda line: len(line) > 0 and not line.isspace(), file_itr)
        mapped_itr = map(self.line_mapper, file_itr)

        return mapped_itr

    def __len__(self):
        if self.length:
            return self.length
        self.length = 0
        logger.info("caculating lines at %s", self.file_path)
        with open(self.file_path, "r") as f:
            for line in f.read().splitlines():
                if len(line) > 0 and not line.isspace():
                    self.length += 1

        return self.length

