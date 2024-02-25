import numpy as np

# Load the data
data_dict = np.load("purchase2_train.npy", allow_pickle=True).item()

# Check if the loaded data is a dictionary
if isinstance(data_dict, dict):
    # Print the shape of the loaded data
    print("Shape of loaded data:")
    print("X:", data_dict["X"].shape)
    print("y:", data_dict["y"].shape)

    # Print the first few elements of the arrays
    print(
        "First few elements of X:", data_dict["X"][:5]
    )  # Adjust the indexing as needed
    print(
        "First few elements of y:", data_dict["y"][:5]
    )  # Adjust the indexing as needed
else:
    print("Loaded data is not a dictionary:", data_dict)
