import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.nn.functional import one_hot
from sharded import sizeOfShard, getShardHash, fetchShardBatch, fetchTestBatch
import os
from glob import glob
from time import time
import json

class SISA:
    def __init__(self, model, dataset, container, shard, slices, label="latest", **kwargs):
        self.model = model
        self.dataset = dataset
        self.container = container
        self.shard = shard
        self.slices = slices
        self.label = label
        self.__dict__.update(kwargs)

    def train(self):
        # Training logic
        pass

    def test(self):
        # Testing logic
        pass

    def _load_dataset_metadata(self):
        # Load dataset metadata
        pass

    def _initialize_model(self):
        # Initialize model
        pass

    def _initialize_optimizer(self):
        # Initialize optimizer
        pass

    def _compute_loss(self):
        # Compute loss
        pass

    def _create_checkpoint(self):
        # Create checkpoint
        pass

    def _load_model_weights(self):
        # Load model weights
        pass

    def _compute_predictions(self):
        # Compute predictions
        pass

if __name__ == "__main__":
    # Your script logic goes here
    pass

