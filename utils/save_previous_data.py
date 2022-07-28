import numpy as np


def save_previous_data(data_list, images, labels, num_samples=None):
    if num_samples is None:
        num_samples = 100
    
    unique_labels = np.unique(labels)

    for class_idx in unique_labels:
        class_idx_array = np.where(labels == class_idx)[0]
        indices = np.random.choice(class_idx_array, num_samples, replace=False)

        for i in indices:
            data_list.append((images[i], labels[i]))
