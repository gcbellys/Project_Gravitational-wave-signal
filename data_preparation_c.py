import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_l1_data(folder, label):
    """
    Load L1 data from the folder and assign labels to each sample.

    Parameters:
        folder (str): Path to the folder containing samples.
        label (int): Class label for the samples (1 for positive, 0 for negative).

    Returns:
        list: A list of (L1_strain, label) samples.
    """
    data_samples = []
    l1_file_count = 0

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # Skip non-directory entries

        l1_file = None

        for file_name in os.listdir(subfolder_path):
            if file_name.endswith("L1_bandpassed_data.txt"):
                l1_file = os.path.join(subfolder_path, file_name)
                l1_file_count += 1
                break  # Only one L1 file is needed per subdirectory

        if l1_file:
            try:
                l1_data = np.loadtxt(l1_file, skiprows=1)[:, 1:]  # Skip first row and column
                if l1_data.ndim == 1:
                    l1_data = l1_data.reshape(-1, 1)
                data_samples.append((l1_data, label))
            except Exception as e:
                print(f"Failed to load L1 file: {l1_file}, Error: {e}")

    print(f"Folder: {folder} -> L1 file count: {l1_file_count}, Loaded samples: {len(data_samples)}")
    return data_samples

def prepare_l1_data(positive_dir, negative_dir, test_ratio=0.1, random_seed=42, pos_neg_ratio=1):
    """
    Prepare training and testing datasets containing only L1 data.

    Parameters:
        positive_dir (str): Path to the positive samples folder.
        negative_dir (str): Path to the negative samples folder.
        test_ratio (float): Proportion of the dataset to include in the test split.
        random_seed (int): Random seed to ensure reproducibility.
        pos_neg_ratio (float): Desired positive-to-negative sample ratio.

    Returns:
        tuple: (training features, testing features, training labels, testing labels)
    """
    positive_samples = load_l1_data(positive_dir, label=1)
    negative_samples = load_l1_data(negative_dir, label=0)

    max_negatives = int(len(positive_samples) * pos_neg_ratio)
    negative_samples = negative_samples[:max_negatives]

    print(f"Number of positive samples: {len(positive_samples)}, Adjusted number of negative samples: {len(negative_samples)}")

    combined_data = positive_samples + negative_samples
    np.random.shuffle(combined_data)

    X_data = [item[0] for item in combined_data]
    y_data = [item[1] for item in combined_data]

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_ratio, random_state=random_seed, stratify=y_data
    )

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"Positive samples in training: {sum(y_train)}, Negative samples in training: {len(y_train) - sum(y_train)}")
    print(f"Positive samples in testing: {sum(y_test)}, Negative samples in testing: {len(y_test) - sum(y_test)}")

    return X_train, X_test, y_train, y_test
