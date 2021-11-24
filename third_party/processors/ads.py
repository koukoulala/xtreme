# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" XNLI utils (dataset loading and evaluation) """


import logging
import os

from transformers import DataProcessor
from .utils import InputExample

logger = logging.getLogger(__name__)


class AdsProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_tsv_examples(self, data_path, language='en', multi_lang=False):
        """See base class."""
        examples = []
        lines = self._read_tsv(data_path)

        for (i, line) in enumerate(lines):
            guid = "%s" % (i)
            if multi_lang:
                text_a = line[0]
                text_b = line[1]
                label = line[2]
            else:
                text_a = line[2]
                text_b = line[3]
                label = line[1]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))

        return examples

    def get_txt_examples(self, data_path, language='en'):
        """See base class."""
        examples = []
        lines = self._read_tsv(data_path)

        for (i, line) in enumerate(lines):
            guid = "%s" % (i)
            text_a = line[2]
            text_b = line[1]
            label = str(line[0].strip())
            if label == "0":
                label_true = "contradiction"
            else:
                label_true = "entailment"
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label_true, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label_true, language=language))

        return examples

    def get_aether_examples(self, data_path, language='en'):
        """See base class."""
        examples = []
        lines = self._read_tsv(data_path)

        for (i, line) in enumerate(lines):
            guid = "%s" % (i)
            id = line[0]
            label = str(line[1].strip())
            desc = line[2]
            chunk = line[3]
            task_id = line[4]
            if label == "0":
                label_true = "contradiction"
            else:
                label_true = "entailment"
            assert isinstance(chunk, str) and isinstance(desc, str) and isinstance(label_true, str)
            examples.append(InputExample(guid=guid, text_a=chunk, text_b=desc, label=label_true, language=language))

        return examples


ads_processors = {
    "ads": AdsProcessor,
}

ads_output_modes = {
    "ads": "classification",
}

ads_tasks_num_labels = {
    "ads": 2,
}
