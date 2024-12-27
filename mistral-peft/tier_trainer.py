import os
import pyvene as pv
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollator,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
from peft.tuners import lora
from transformers.trainer_utils import EvalPrediction, has_length, denumpify_detensorize
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from tqdm import tqdm
from functools import reduce
import os
import torch
import re
import evaluate
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from transformers.trainer_pt_utils import get_parameter_names

logger = logging.get_logger(__name__)



def make_dataloader(
    dataset: Dataset, batch_size: int, collate_fn: DataCollatorForSeq2Seq, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn
    )

class TierTrainer(Trainer):

    # 重写该方法，防止删除数据中新增的列
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(
            self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True
        )