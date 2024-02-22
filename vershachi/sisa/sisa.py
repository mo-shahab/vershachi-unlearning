import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.nn.functional import one_hot
from .sharded import sizeOfShard, getShardHash, fetchShardBatch
import os
from glob import glob
from time import time
from importlib import import_module
import json
import sys  

class SisaTrainer:
    def __init__(self, args, model_dir):
        self.args = args
        self.model_dir = model_dir
        self.model_lib = self._import_model_module()
        self.loss_fn = CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()

    def _import_model_module(self):
        sys.path.append(self.model_dir)  # Add the model directory to the module search path
        model_module_name = self.args.model
        model_lib = import_module(model_module_name)
        return model_lib

    def _init_model(self):
        model = self.model_lib.Model(input_shape, nb_classes, dropout_rate=self.args.dropout_rate)
        model.to(self.device)
        return model

    def _init_optimizer(self):
        if self.args.optimizer == "adam":
            optimizer = Adam(model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == "sgd":
            optimizer = SGD(model.parameters(), lr=self.args.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")
        return optimizer

    def train(self):
        if self.args.train:
            shard_size = sizeOfShard(self.args.container, self.args.shard)
            slice_size = shard_size // self.args.slices
            avg_epochs_per_slice = (
                2 * self.args.slices / (self.args.slices + 1) * self.args.epochs / self.args.slices
            )
            loaded = False

            for sl in range(self.args.slices):
                # Get slice hash using sharded lib.
                slice_hash = getShardHash(
                    self.args.container, self.args.label, self.args.shard, until=(sl + 1) * slice_size
                )

                # Initialize state.
                elapsed_time = 0
                start_epoch = 0
                slice_epochs = int((sl + 1) * avg_epochs_per_slice) - int(
                    sl * avg_epochs_per_slice
                )

                # If weights are already in memory (from previous slice), skip loading.
                if not loaded:
                    # Look for a recovery checkpoint for the slice.
                    recovery_list = glob(
                        f"containers/{self.args.container}/cache/{slice_hash}_*.pt"
                    )
                    if len(recovery_list) > 0:
                        print(
                            f"Recovery mode for shard {self.args.shard} on slice {sl}"
                        )

                        # Load weights.
                        self.model.load_state_dict(torch.load(recovery_list[0]))
                        start_epoch = int(
                            recovery_list[0].split("/")[-1].split(".")[0].split("_")[1]
                        )

                        # Load time
                        with open(
                            f"containers/{self.args.container}/times/{slice_hash}_{start_epoch}.time",
                            "r",
                        ) as f:
                            elapsed_time = float(f.read())

                    # If there is no recovery checkpoint and this slice is not the first, load previous slice.
                    elif sl > 0:
                        previous_slice_hash = getShardHash(
                            self.args.container, self.args.label, self.args.shard, until=sl * slice_size
                        )

                        # Load weights.
                        self.model.load_state_dict(
                            torch.load(
                                f"containers/{self.args.container}/cache/{previous_slice_hash}.pt"
                            )
                        )

                    # Mark model as loaded for next slices.
                    loaded = True

                # Actual training.
                train_time = 0.0

                for epoch in range(start_epoch, slice_epochs):
                    epoch_start_time = time()

                    for images, labels in fetchShardBatch(
                        self.args.container,
                        self.args.label,
                        self.args.shard,
                        self.args.batch_size,
                        self.args.dataset,
                        until=(sl + 1) * slice_size if sl < self.args.slices - 1 else None,
                    ):

                        # Convert data to torch format and send to selected device.
                        gpu_images = torch.from_numpy(images).to(
                            self.device
                        )  
                        gpu_labels = torch.from_numpy(labels).to(
                            self.device
                        )

                        forward_start_time = time()

                        # Perform basic training step.
                        logits = self.model(gpu_images)
                        loss = self.loss_fn(logits, gpu_labels)

                        self.optimizer.zero_grad()
                        loss.backward()

                        self.optimizer.step()

                        train_time += time() - forward_start_time

                    # Save weights and time for every epoch.
                    torch.save(
                        self.model.state_dict(),
                        f"containers/{self.args.container}/cache/{slice_hash}_{epoch}.pt"
                    )
                    with open(
                        f"containers/{self.args.container}/times/{slice_hash}_{epoch}.time",
                        "w",
                    ) as f:
                        f.write("{}\n".format(train_time + elapsed_time))

                    # Remove previous checkpoint.
                    if epoch > 0:
                        os.remove(
                            f"containers/{self.args.container}/cache/{slice_hash}_{epoch - 1}.pt"
                        )
                        os.remove(
                            f"containers/{self.args.container}/times/{slice_hash}_{epoch - 1}.time"
                        )

                # If this is the last slice, create a final checkpoint.
                if sl == self.args.slices - 1:
                    os.rename(
                        f"containers/{self.args.container}/cache/{slice_hash}_{slice_epochs - 1}.pt",
                        f"containers/{self.args.container}/cache/shard-{self.args.shard}_{self.args.label}.pt"
                    )
                    os.rename(
                        f"containers/{self.args.container}/times/{slice_hash}_{slice_epochs - 1}.time",
                        f"containers/{self.args.container}/times/shard-{self.args.shard}_{self.args.label}.time"
                    )

