import random

# Assuming your data is stored in a dictionary called 'data'
# with keys 'x' and 'y'
data = {
    "x": ...,  # Your feature data
    "y": ...,  # Your label data
}

# Define the size of the subset you want to create
subset_size = 1000  # Adjust this as needed

# Get the total number of samples in your dataset
total_samples = len(data["x"])

# Generate random indices to select samples for the subset
subset_indices = random.sample(range(total_samples), subset_size)

# Create the subset
subset_data = {
    "x": [data["x"][i] for i in subset_indices],
    "y": [data["y"][i] for i in subset_indices],
}

# Now 'subset_data' contains a subset of your original data
