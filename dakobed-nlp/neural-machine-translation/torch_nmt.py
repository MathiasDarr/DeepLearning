from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from model_embeddings import ModelEmbeddings
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.module):
    pass