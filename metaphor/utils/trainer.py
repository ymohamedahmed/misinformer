
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, lr: float, patience: int):
        pass

    def fit(self, model: nn.Module, train: DataLoader, validation: DataLoader):
        pass
    