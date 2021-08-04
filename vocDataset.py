# 2021-04-22 Pytorch transformers sentence classification
# -*- coding: utf-8 -*-

import sys
import torch
import pandas as pd
from torch.utils.data import Dataset

# data set control
class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return text, label

