"""
script is broken, freezes the pc when executed

"""

import numpy as np

def create_subset(data, subset_size):
    indices = data['indices']
    indptr = data['indptr']
    total_data_points = indptr.shape[0] - 1

    if subset_size >= total_data_points:
        return data

    # Randomly select subset indices
    subset_indices = np.random.choice(total_data_points, subset_size, replace=False)

    # Extract corresponding data using subset indices
    subset_indptr = indptr[subset_indices]
    subset_indptr = np.append(subset_indptr, len(indices))  # Add the end index

    # Calculate subset indices for the indices array
    subset_indices_data = np.concatenate([np.arange(start, end) for start, end in zip(subset_indptr[:-1], subset_indptr[1:])])

    # Create subset data
    subset_data = {
        'indices': indices[subset_indices_data],
        'indptr': np.cumsum([0] + [end - start for start, end in zip(subset_indptr[:-1], subset_indptr[1:])]),
        'format': data['format'],
        'shape': (subset_size, data['shape'][1]),  # Assuming shape[0] is the number of rows
        'data': data['data'][subset_indices_data]
    }

    return subset_data

def main():
    input_files = ["data1.npz", "data2.npz"]  # Update with your NPZ file paths
    output_files = ["subset_data1.npz", "subset_data2.npz"]  # Update with desired output file paths
    subset_size = 1000  # Update with desired subset size

    for input_file, output_file in zip(input_files, output_files):
        data = np.load(input_file)
        subset_data = create_subset(data, subset_size)
        np.savez(output_file, **subset_data)
        print(f"Subset data saved to {output_file}")

if __name__ == "__main__":
    main()

