from typing import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.linalg as linalg

from tqdm import tqdm


def default(val, d):
    return val if val is not None else d