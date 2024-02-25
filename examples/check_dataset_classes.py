import numpy as np

# Load your data
train_data = np.load("purchase2_train.npy", allow_pickle=True)
test_data = np.load("purchase2_test.npy", allow_pickle=True)

# Extract the target labels
y_train = train_data.item().get('y')
y_test = test_data.item().get('y')

# Concatenate train and test labels to ensure you capture all unique values
all_labels = np.concatenate([y_train, y_test])

# Find unique classes
unique_classes = np.unique(all_labels)
num_classes = len(unique_classes)
print("Number of classes:", num_classes)
print("Unique classes:", unique_classes)

