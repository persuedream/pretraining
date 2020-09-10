import csv
import glob
import os
from typing import List

import tqdm
from transformers import InputExample, DataProcessor


class LongformerProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        self._create_examples(self._get_text(os.path.join(data_dir, "train.(json|txt|csv)")))

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        self._create_examples(self._get_text(os.path.join(data_dir, "valid.(json|txt|csv)")))

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        self._create_examples(self._get_text(os.path.join(data_dir, "test.(json|txt|csv")))

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _get_text(self, data_dir):
        """
        根据文件路径获取文本
        :param data_dir:
        :return:
        """
        lines = []
        files = glob.glob(data_dir, recursive=True)
        for f in tqdm.tqdm(files, desc="read files"):
            with open(f, "r", encoding="utf-8") as fin:
                data_raw = csv.reader(fin)
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines: List[List[str]]):
        """Creates examples for the training and dev sets."""
        examples = [
            InputExample(
                example_id=ind,
                text_a=line,
                text_b=None,

            )
            for ind, line in enumerate(lines)  # we skip the line with the column names
        ]

        return examples

processors = {
    "LongformerProcessor": LongformerProcessor
}