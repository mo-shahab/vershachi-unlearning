"""
Script Description:
This script provides functions for managing shards, fetching batches of data from shards, and fetching test batches.

"""

import numpy as np
from hashlib import sha256
import importlib
import json
import os
import importlib.util


def sizeOfShard(container, shard):
    """
    Returns the size (in number of points) of the shard before any unlearning request.

    Parameters:
        - container (str): Path to the container directory.
        - shard (int): Shard index.

    Returns:
        - int: Size of the shard.
    """
    shards = np.load("containers/{}/splitfile.npy".format(container), allow_pickle=True)

    return shards[shard].shape[0]


def realSizeOfShard(container, label, shard):
    """
    Returns the actual size of the shard (including unlearning requests).

    Parameters:
        - container (str): Path to the container directory.
        - label (str): Label for the outputs.
        - shard (int): Shard index.

    Returns:
        - int: Actual size of the shard.
    """
    shards = np.load("containers/{}/splitfile.npy".format(container), allow_pickle=True)
    requests = np.load(
        "containers/{}/requestfile:{}.npy".format(container, label), allow_pickle=True
    )

    return shards[shard].shape[0] - requests[shard].shape[0]


def getShardHash(container, label, shard, until=None):
    """
    Returns a hash of the indices of the points in the shard lower than until
    that are not in the requests (separated by :).

    Parameters:
        - container (str): Path to the container directory.
        - label (str): Label for the outputs.
        - shard (int): Shard index.
        - until (int): Upper limit for the indices (default is None).

    Returns:
        - str: Hash of the indices.
    """
    shards = np.load("containers/{}/splitfile.npy".format(container), allow_pickle=True)
    requests = np.load(
        "containers/{}/requestfile_{}.npy".format(container, label), allow_pickle=True
    )

    if until == None:
        until = shards[shard].shape[0]
    indices = np.setdiff1d(shards[shard][:until], requests[shard])
    string_of_indices = ":".join(indices.astype(str))
    return sha256(string_of_indices.encode()).hexdigest()


# def fetchShardBatch(container, label, shard, batch_size, dataset, offset=0, until=None):
#     '''
#     Generator returning batches of points in the shard that are not in the requests
#     with specified batch_size from the specified dataset
#     optionnally located between offset and until (slicing).
#     '''
#     shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
#     requests = np.load('containers/{}/requestfile_{}.npy'.format(container, label), allow_pickle=True)

#     with open(dataset) as f:
#         datasetfile = json.loads(f.read())
#     dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))
#     if until == None or until > shards[shard].shape[0]:
#         until = shards[shard].shape[0]

#     limit = offset
#     while limit <= until - batch_size:
#         limit += batch_size
#         indices = np.setdiff1d(shards[shard][limit-batch_size:limit], requests[shard])
#         yield dataloader.load(indices)
#     if limit < until:
#         indices = np.setdiff1d(shards[shard][limit:until], requests[shard])
#         yield dataloader.load(indices)


def fetchShardBatch(container, label, shard, batch_size, dataset, offset=0, until=None):
    """
    Generator returning batches of points in the shard that are not in the requests
    with specified batch_size from the specified dataset
    optionally located between offset and until (slicing).

    Parameters:
        - container (str): Path to the container directory.
        - label (str): Label for the outputs.
        - shard (int): Shard index.
        - batch_size (int): Size of each batch.
        - dataset (str): Location of the dataset file.
        - offset (int): Starting index for slicing (default is 0).
        - until (int): Ending index for slicing (default is None).

    Yields:
        - np.ndarray: Batch of data.
    """
    shards = np.load("containers/{}/splitfile.npy".format(container), allow_pickle=True)
    requests = np.load(
        "containers/{}/requestfile_{}.npy".format(container, label), allow_pickle=True
    )

    with open(dataset) as f:
        datasetfile = json.loads(f.read())

    # Get the directory containing the datasetfile
    dataset_dir = os.path.dirname(dataset)
    # Construct the path to the dataloader module
    dataloader_path = os.path.join(dataset_dir, datasetfile["dataloader"] + ".py")

    # Load the dataset module
    spec = importlib.util.spec_from_file_location(
        datasetfile["dataloader"], dataloader_path
    )
    dataloader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataloader_module)

    if until is None or until > shards[shard].shape[0]:
        until = shards[shard].shape[0]

    limit = offset
    while limit <= until - batch_size:
        limit += batch_size
        indices = np.setdiff1d(
            shards[shard][limit - batch_size : limit], requests[shard]
        )
        yield dataloader_module.load(indices)
    if limit < until:
        indices = np.setdiff1d(shards[shard][limit:until], requests[shard])
        yield dataloader_module.load(indices)


# def fetchTestBatch(dataset, batch_size):
#     '''
#     Generator returning batches of points from the specified test dataset
#     with specified batch_size.
#     '''
#     with open(dataset) as f:
#         datasetfile = json.loads(f.read())
#     dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))

#     limit = 0
#     while limit <= datasetfile['nb_test'] - batch_size:
#         limit += batch_size
#         yield dataloader.load(np.arange(limit - batch_size, limit), category='test')
#     if limit < datasetfile['nb_test']:
#         yield dataloader.load(np.arange(limit, datasetfile['nb_test']), category='test')


def fetchTestBatch(dataset, batch_size):
    """
    Generator returning batches of points from the specified test dataset
    with specified batch_size.

    Parameters:
        - dataset (str): Location of the dataset file.
        - batch_size (int): Size of each batch.

    Yields:
        - np.ndarray: Batch of test data.
    """
    with open(dataset) as f:
        datasetfile = json.loads(f.read())

    # Get the directory containing the dataset file
    dataset_dir = os.path.dirname(dataset)
    # Get the dataset module name from datasetfile["dataloader"]
    dataset_module_name = datasetfile["dataloader"]
    # Construct the path to the dataset file
    dataset_file_path = os.path.join(dataset_dir, f"{dataset_module_name}.py")

    # Load the dataset module from the specified path
    spec = importlib.util.spec_from_file_location(
        dataset_module_name, dataset_file_path
    )
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)

    limit = 0
    while limit <= datasetfile["nb_test"] - batch_size:
        limit += batch_size
        yield dataset_module.load(np.arange(limit - batch_size, limit), category="test")
    if limit < datasetfile["nb_test"]:
        yield dataset_module.load(
            np.arange(limit, datasetfile["nb_test"]), category="test"
        )
