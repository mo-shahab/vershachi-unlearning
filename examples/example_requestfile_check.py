import numpy as np

# Load the request file
requests = np.load("containers/default/requestfile_latest.npy")

# load the split file
splitfile = np.load("containers/default/splitfile.npy")

# Inspect the shape and contents of the requests array
print("Shape of requests array:", requests.shape)
print("Contents of requests array:", requests)

print("\nshape of splitfile:", splitfile.shape)
print("contents of splitfile:", splitfile)
