from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()
        # Initialize any layers or parameters here if needed
        

    @abstractmethod
    def train(self, mode=True):
        pass

    @abstractmethod
    def train_with_k_fold(self, k: int = 5, seed: int = None, shuffle: bool = True):
        pass