import os
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz


# Loads dataset
def load_data(file_paths):
    if isinstance(file_paths, str) and file_paths.endswith(".npz"):
        # If the input is a single NPZ file path
        data = load_npz(file_paths).toarray().astype(int)
    elif isinstance(file_paths, list) and all(fp.endswith(".npz") for fp in file_paths):
        # If the input is a list of NPZ file paths
        data = np.concatenate(
            [load_npz(file_path).toarray() for file_path in file_paths]
        ).astype(int)
    elif isinstance(file_paths, str) and file_paths.endswith(".json"):
        # If the input is a JSON file containing metadata with file paths
        with open(file_paths) as f:
            metadata = json.load(f)
        data_paths = metadata.get("data_paths", [])
        data = np.concatenate(
            [load_npz(file_path).toarray() for file_path in data_paths]
        ).astype(int)
    elif isinstance(file_paths, str):
        # If file_paths is a string, assume it's a path to a JSON file containing metadata.
        with open(file_paths) as f:
            metadata = json.load(f)
        # Return the metadata directly.
        return metadata
    else:
        raise ValueError("Unsupported input format or file type")
    return data


# Uses K-Means algorithm to make clusters
def apply_kmeans(data, num_clusters=2, random_state=0):
    kmeans_file = f"{num_clusters}_kmeans.npy"
    if not os.path.exists(kmeans_file):
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state).fit(data)
        labels = kmeans.labels_
        np.save(kmeans_file, labels)
    else:
        labels = np.load(kmeans_file)
    return labels


# Splits the data, resulting in two parts, test data and train data
def split_data(data, labels, test_size=0.2, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# Writes the data which splits into the numpy binaries
def save_split_data(
    X_train, X_test, y_train, y_test, num_clusters=2, prefix="purchase", output_dir=None
):
    if output_dir is None:
        output_dir = (
            os.getcwd()
        )  # Use the current working directory if output_dir is not specified

    train_file = os.path.join(output_dir, f"{prefix}{num_clusters}_train.npy")
    test_file = os.path.join(output_dir, f"{prefix}{num_clusters}_test.npy")

    np.save(train_file, {"X": X_train, "y": y_train})
    np.save(test_file, {"X": X_test, "y": y_test})
