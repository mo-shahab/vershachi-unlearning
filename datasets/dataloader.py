import numpy as np
import os

# pwd = os.path.dirname(os.path.realpath(__file__))
# directory of the stuff where the training and testing things were there

# train_data = np.load(os.path.join(dir, 'purchase2_train.npy'), allow_pickle=True)
# test_data = np.load(os.path.join(dir, 'purchase2_test.npy'), allow_pickle=True)

train_data = np.load(
    r"C:\dev\vershachi-unlearning\examples\purchase2_train.npy", allow_pickle=True
)
test_data = np.load(
    r"C:\dev\vershachi-unlearning\examples\purchase2_test.npy", allow_pickle=True
)

train_data = train_data.reshape((1,))[0]
test_data = test_data.reshape((1,))[0]

X_train = train_data["X"].astype(np.float32)
X_test = test_data["X"].astype(np.float32)
y_train = train_data["y"].astype(np.int64)
y_test = test_data["y"].astype(np.int64)


def load(indices, category="train"):
    if category == "train":
        return X_train[indices], y_train[indices]
    elif category == "test":
        return X_test[indices], y_test[indices]
