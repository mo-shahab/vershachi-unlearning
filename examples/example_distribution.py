import os
import numpy as np
import json
from scipy.sparse import load_npz
from vershachi.sisa.preprocessing import load_data
from vershachi.sisa.distribution import split_dataset, generate_and_distribute_requests

# Paths and parameters
datasetfile = "../datasets/datasetfile"
container = "default"
# label = "latest"

# Load data
data = load_data(datasetfile)

# Split dataset
shards = 4 # Change this to the desired number of shards
distribution = "uniform"  # Change this to the desired distribution
split_dataset(shards, distribution, container, datasetfile, label="0")

# Generate and distribute requests
# num_requests = 1000  # Change this to the desired number of requests

# path to the splitfile.npy
partition = np.load("./containers/default/splitfile.npy")
# print(partition)

# Loop to generate and distribute requests for different numbers of shards
for j in range(1, 16):
    num_requests = int(j * shards / 5)
    # partition = np.load("./containers/default/splitfile.npy")
    label = "" + str(j)
    generate_and_distribute_requests(
        num_requests, distribution, container, label, partition, datasetfile
    )
