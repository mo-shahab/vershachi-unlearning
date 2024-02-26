import numpy as np


def print_data_info(data):
    for key, value in data.items():
        print(f"Key: {key}")
        if isinstance(value, np.ndarray):
            print(f"Shape: {value.shape}")
            print("Data:")
            print(value)
        else:
            print("Data:")
            print(value)
        print("\n")


# Assuming data1 and data2 are your loaded npz files
data1 = np.load("data1.npz")
data2 = np.load("data2.npz")

print("Data from data1.npz:")
print_data_info(data1)

print("Data from data2.npz:")
print_data_info(data2)
