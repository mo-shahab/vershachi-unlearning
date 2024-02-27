import pandas as pd

def compute_time_stats(container):
    """
    Compute statistics based on the execution time (cumulated feed-forward + backprop.) of the shards.

    Parameters:
        - container (str): Name of the container.

    Returns:
        - tuple: A tuple containing the sum and mean of the execution times.
    """
    # Read the execution times from the CSV file
    times_df = pd.read_csv(f"containers/{container}/times/times.tmp", names=["time"])
    
    # Compute sum and mean of the execution times
    total_time = times_df["time"].sum()
    mean_time = times_df["time"].mean()
    
    return total_time, mean_time

