import numpy as np
import json
import os


def split_dataset(shards, distribution, container, dataset, label="latest"):
    # Load dataset metadata.
    with open(dataset) as f:
        datasetfile = json.load(f)

    if shards is not None:
        # If distribution is uniform, split without optimizing.
        if distribution == "uniform":
            partition = np.split(
                np.arange(0, datasetfile["nb_train"]),
                [t * (datasetfile["nb_train"] // shards) for t in range(1, shards)],
            )

            # Create directories if they don't exist
            save_dir = f"containers/{container}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save partition to a numpy file.
            np.save(f"{save_dir}/splitfile.npy", partition)

            # Create empty request files for each shard.
            requests = [np.array([]) for _ in range(shards)]
            np.save(f"{save_dir}/requestfile:{label}.npy", requests)


def generate_requests(num_requests, distribution, datasetfile):
    if distribution.split(":")[0] == "exponential":
        lbd = (
            float(distribution.split(":")[1])
            if len(distribution.split(":")) > 1
            else -np.log(0.05) / datasetfile["nb_train"]
        )
        return np.random.exponential(1 / lbd, (num_requests,))
    elif distribution.split(":")[0] == "pareto":
        a = (
            float(distribution.split(":")[1])
            if len(distribution.split(":")) > 1
            else 1.16
        )
        return np.random.pareto(a, (num_requests,))
    else:
        return np.random.randint(0, datasetfile["nb_train"], num_requests)


def generate_and_distribute_requests(
    requests, distribution, container, label, partition, datasetfile
):
    if requests is not None:
        if distribution == "reset":
            # Reset request files.
            requests = [np.array([]) for _ in range(partition.shape[0])]
            # np.save(f"containers/{container}/requestfile:{label}.npy", requests)
            request_path = os.path.join(
                "containers", container, f"requestfile:{label}.npy"
            )
            np.save(request_path, requests)
        else:
            # Generate unlearning requests.
            all_requests = generate_requests(requests, distribution, datasetfile)

            # Distribute requests among shards.
            requests = distribute_requests(partition, all_requests)

            # Save distributed requests.
            # np.save(f"containers/{container}/requestfile:{label}.npy", requests)
            request_path = os.path.join(
                "containers", container, f"requestfile:{label}.npy"
            )
            np.save(request_path, requests)


def distribute_requests(partition, all_requests):
    requests = [np.intersect1d(part, all_requests) for part in partition]

    # Pad requests to ensure consistent length
    max_length = max(len(r) for r in requests)
    padded_requests = [
        np.pad(r, (0, max_length - len(r)), mode="constant") for r in requests
    ]

    return padded_requests
