import os
import pandas as pd
from vershachi.sisa.aggregation import compute_aggregation_stats
from vershachi.sisa.time import compute_time_stats

shards = 3  # change the number of shard accordingly

# Create general-report.csv if it doesn't exist
if not os.path.exists("general-report.csv"):
    with open("general-report.csv", "w") as report_file:
        report_file.write("nb_shards,nb_requests,accuracy,retraining_time\n")

dataset_file = r"C:\dev\vershachi-unlearning\datasets\datasetfile"

# Loop through shard requests
for j in range(5):  # Iterating 16 times as in the original shell script
    r = j * shards // 5
    try:
        votes, acc = compute_aggregation_stats(
            strategy="uniform",
            container="containers\\" + str(shards),
            shards=shards,
            dataset=dataset_file,
            label=str(r),
        )
        os.system(
            f"cat containers/{shards}/times/shard-*_{r}.time > containers/{shards}/times/times"
        )
        total_time, mean_time = compute_time_stats(container=shards)

        # Append results to general-report.csv
        with open("general-report.csv", "a") as report_file:
            report_file.write(f"{shards},{r},{acc},{total_time}\n")

        print(f"Accuracy: {acc}, Total Time: {total_time}, Mean Time: {mean_time}")
    except FileNotFoundError as e:
        # Ignore errors related to the last shards
        if r == shards - 1:
            continue
        print(f"File not found for shard {r}. Skipping...")
        print(f"Error: {e}")
    except Exception as e:
        # Print any other exceptions for debugging purposes
        print(f"An error occurred for shard {r}: {e}")
