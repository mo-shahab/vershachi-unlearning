import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.nn.functional import one_hot
from .sharded import sizeOfShard, getShardHash, fetchShardBatch, fetchTestBatch
import os
from glob import glob
from time import time
from importlib import import_module
import json
import sys


class SisaTrainer:
    def __init__(
        self,
        model_dir,
        dataset_file,
        model_name="purchase",
        train=True,
        test=False,
        epochs=20,
        batch_size=16,
        dropout_rate=0.4,
        learning_rate=0.001,
        optimizer="sgd",
        output_type="argmax",
        container="default",
        shard=2,
        slices=1,
        chkpt_interval=1,
        label="latest",
    ):
        self.model_dir = model_dir
        self.dataset_file = dataset_file
        self.model_name = model_name
        self.train = train
        self.test = test
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.output_type = output_type
        self.container = container
        self.shard = shard
        self.slices = slices
        self.chkpt_interval = chkpt_interval
        self.label = label

        self.input_shape, self.nb_classes = self._get_dataset_metadata()
        self.model_lib = self._import_model_module()
        self.loss_fn = CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()

    def _import_model_module(self):
        sys.path.append(self.model_dir)
        model_lib = import_module(f"{self.model_name}")
        return model_lib

    def _get_dataset_metadata(self):
        # Construct the absolute path to the dataset file
        # dataset_file_path = os.path.join(self.dataset_dir, self.dataset_file)
        with open(self.dataset_file) as f:
            dataset_metadata = json.load(f)
        input_shape = tuple(dataset_metadata["input_shape"])
        nb_classes = dataset_metadata["nb_classes"]
        return input_shape, nb_classes

    def _init_model(self):
        # Pass input_shape and nb_classes to _init_model()
        model = self.model_lib.Model(
            self.input_shape, self.nb_classes, dropout_rate=self.dropout_rate
        )
        model.to(self.device)
        return model

    def _init_optimizer(self):
        if self.optimizer == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Unsupported optimizer")
        return optimizer

    def _train(self):
        if self.train:
            shard_size = sizeOfShard(self.container, self.shard)
            slice_size = shard_size // self.slices
            avg_epochs_per_slice = (
                2 * self.slices / (self.slices + 1) * self.epochs / self.slices
            )
            loaded = False

            for sl in range(self.slices):
                # Get slice hash using sharded lib.
                slice_hash = getShardHash(
                    self.container, self.label, self.shard, until=(sl + 1) * slice_size
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
                        f"containers/{self.container}/cache/{slice_hash}_*.pt"
                    )
                    if len(recovery_list) > 0:
                        print(f"Recovery mode for shard {self.shard} on slice {sl}")

                        # Load weights.
                        self.model.load_state_dict(torch.load(recovery_list[0]))
                        start_epoch = int(
                            recovery_list[0].split("/")[-1].split(".")[0].split("_")[1]
                        )

                        # Load time
                        with open(
                            f"containers/{self.container}/times/{slice_hash}_{start_epoch}.time",
                            "r",
                        ) as f:
                            elapsed_time = float(f.read())

                    # If there is no recovery checkpoint and this slice is not the first, load previous slice.
                    elif sl > 0:
                        previous_slice_hash = getShardHash(
                            self.container,
                            self.label,
                            self.shard,
                            until=sl * slice_size,
                        )

                        # Load weights.
                        self.model.load_state_dict(
                            torch.load(
                                f"containers/{self.container}/cache/{previous_slice_hash}.pt"
                            )
                        )

                    # Mark model as loaded for next slices.
                    loaded = True

                # Actual training.
                train_time = 0.0

                for epoch in range(start_epoch, slice_epochs):
                    epoch_start_time = time()

                    for images, labels in fetchShardBatch(
                        self.container,
                        self.label,
                        self.shard,
                        self.batch_size,
                        self.dataset_file,
                        until=(sl + 1) * slice_size if sl < self.slices - 1 else None,
                    ):

                        # Convert data to torch format and send to selected device.
                        gpu_images = torch.from_numpy(images).to(self.device)
                        gpu_labels = torch.from_numpy(labels).to(self.device)

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
                        f"containers/{self.container}/cache/{slice_hash}_{epoch}.pt",
                    )
                    with open(
                        f"containers/{self.container}/times/{slice_hash}_{epoch}.time",
                        "w",
                    ) as f:
                        f.write("{}\n".format(train_time + elapsed_time))

                    # Remove previous checkpoint.
                    if epoch > 0:
                        previous_checkpoint_path = f"containers/{self.container}/cache/{slice_hash}_{epoch - 1}.pt"
                        previous_time_file_path = f"containers/{self.container}/times/{slice_hash}_{epoch - 1}.time"

                        # Check if the files exist before attempting to remove them
                        if os.path.exists(previous_checkpoint_path):
                            try:
                                os.remove(previous_checkpoint_path)
                                print(
                                    f"Previous checkpoint removed: {previous_checkpoint_path}"
                                )
                            except Exception as e:
                                print(f"Error removing previous checkpoint: {e}")

                        if os.path.exists(previous_time_file_path):
                            try:
                                os.remove(previous_time_file_path)
                                print(
                                    f"Previous time file removed: {previous_time_file_path}"
                                )
                            except Exception as e:
                                print(f"Error removing previous time file: {e}")

                # If this is the last slice, create a final checkpoint.
                if sl == self.slices - 1:
                    destination_checkpoint_path = f"containers/{self.container}/cache/shard-{self.shard}_{self.label}.pt"
                    destination_time_file_path = f"containers/{self.container}/times/shard-{self.shard}_{self.label}.time"

                    if os.path.exists(destination_checkpoint_path):
                        os.remove(destination_checkpoint_path)

                    if os.path.exists(destination_time_file_path):
                        os.remove(destination_time_file_path)

                    os.rename(
                        f"containers/{self.container}/cache/{slice_hash}_{slice_epochs - 1}.pt",
                        destination_checkpoint_path,
                    )
                    os.rename(
                        f"containers/{self.container}/times/{slice_hash}_{slice_epochs - 1}.time",
                        destination_time_file_path,
                    )

    def _test(self):
        if self.test:
            # Load model weights from shard checkpoint (last slice).
            checkpoint_path = (
                f"containers/{self.container}/cache/shard-{self.shard}_{self.label}.pt"
            )
            self.model.load_state_dict(torch.load(checkpoint_path))

            # Compute predictions batch per batch.
            outputs = np.empty((0, self.nb_classes))
            for images, _ in fetchTestBatch(self.dataset_file, self.batch_size):
                # Convert data to torch format and send to selected device.
                gpu_images = torch.from_numpy(images).to(self.device)

                if self.output_type == "softmax":
                    # Actual batch prediction.
                    logits = self.model(gpu_images)
                    predictions = torch.softmax(logits, dim=1).cpu().numpy()

                else:
                    # Actual batch prediction.
                    logits = self.model(gpu_images)
                    predictions = torch.argmax(logits, dim=1)  # Get class indices
                    predictions = torch.nn.functional.one_hot(
                        predictions, self.nb_classes
                    )  # Convert to one-hot tensor

                # Concatenate with previous batches.
                outputs = np.concatenate((outputs, predictions))

            # Save outputs in numpy format.
            output_file_path = f"containers/{self.container}/outputs/shard-{self.shard}_{self.label}.npy"
            np.save(output_file_path, outputs)
