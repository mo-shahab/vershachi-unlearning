import numpy as np

# Load the training and testing data
train_data_path = r"C:\dev\vershachi-unlearning\examples\purchase2_train.npy"
test_data_path = r"C:\dev\vershachi-unlearning\examples\purchase2_test.npy"
train_data = np.load(train_data_path, allow_pickle=True).item()
test_data = np.load(test_data_path, allow_pickle=True).item()

# Extract features and labels from train_data
X_train = train_data.get("X").astype(np.float32)
y_train = train_data.get("y").astype(np.int64)

# Extract features and labels from test_data
X_test = test_data.get("X").astype(np.float32)
y_test = test_data.get("y").astype(np.int64)


# Define a function to load data based on indices
def load(indices, category="train"):
    if category == "train":
        return X_train[indices], y_train[indices]
    elif category == "test":
        return X_test[indices], y_test[indices]
