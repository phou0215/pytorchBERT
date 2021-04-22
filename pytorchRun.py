# 2021-04-22 Pytorch transformers sentence classification
# -*- coding: utf-8 -*-

import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import optimizer, Adam
from torch.nn import functional as F

rand = torch.rand(5,3)
print(rand)