from vershachi.sisa.preprocessing import (
    load_data,
    apply_kmeans,
    split_data,
    save_split_data,
)

# Example usage
file_paths = ["../datasets/data1.npz", "../datasets/data2.npz"]
data = load_data(file_paths)
labels = apply_kmeans(data, num_clusters=5)  # Adjust num_clusters as needed
X_train, X_test, y_train, y_test = split_data(data, labels)
save_split_data(X_train, X_test, y_train, y_test)
