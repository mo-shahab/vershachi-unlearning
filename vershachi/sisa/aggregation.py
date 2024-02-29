"""
Script Description:
This script defines a function to aggregate outputs from different shards based on a specified strategy.
The aggregation process combines the outputs from multiple shards into a single aggregated result,
which can be used in various machine learning tasks such as federated learning.
"""

import numpy as np
import json
import os


def compute_aggregation_stats(
    strategy, container, shards, dataset, label, baseline=None
):
    """
    Aggregate outputs from different shards based on the specified strategy.

    Parameters:
        - strategy (str): The voting strategy ('uniform', 'models:', 'proportional').
        - container (str): Path to the container directory.
        - shards (int): Number of shards.
        - dataset (str): Location of the dataset file.
        - label (str): Label for the outputs.
        - baseline (int): Use only the specified shard as a lone shard baseline.

    Returns:
        - np.ndarray: Aggregated votes.
    """
    # Load dataset metadata
    with open(dataset) as f:
        datasetfile = json.load(f)

    # Output files used for the vote
    if baseline is not None:
        filenames = [f"shard-{baseline}_{label}.npy"]
    else:
        filenames = [f"shard-{i}_{label}.npy" for i in range(shards)]

    # Concatenate output files
    outputs = []
    for filename in filenames:
        outputs.append(
            np.load(os.path.join(container, "outputs", filename), allow_pickle=True)
        )

    outputs = np.array(outputs)

    # Compute weight vector based on given strategy
    if strategy == "uniform":
        weights = np.ones(outputs.shape[0]) / outputs.shape[0]
    elif strategy.startswith("models:"):
        models = np.array(strategy.split(":")[1].split(",")).astype(int)
        weights = np.zeros(outputs.shape[0])
        weights[models] = 1 / models.shape[0]
    elif strategy == "proportional":
        split = np.load(os.path.join(container, "splitfile.npy"), allow_pickle=True)
        weights = np.array([shard.shape[0] for shard in split])

    # Tensor contraction of outputs and weights (on the shard dimension)
    votes = np.argmax(
        np.tensordot(weights.reshape(1, -1), outputs, axes=1), axis=2
    ).reshape(outputs.shape[1])

    return votes
