import numpy as np

# Load the data
data1 = np.load("data1.npz")
data2 = np.load("data2.npz")

# Inspect the keys of the data
print("Keys of data1.npz:", data1.keys())
print("Keys of data2.npz:", data2.keys())
