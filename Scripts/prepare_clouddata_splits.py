# prepare_clouddata_splits.py

import os
import sys
import time
import contextlib
import pickle
import random
import numpy as np
import pandas as pd
import mlstac
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Parameters for splits
NUM_TRAIN = 1000  # number of training images
NUM_VAL   = 200   # number of validation images
NUM_TEST  = 100   # number of test images
TOTAL_REQUIRED = NUM_TRAIN + NUM_VAL + NUM_TEST  # total required images

# Start timer
print("=== Starting dataset download and split preparation ===")
start_time = time.time()

# Load dataset via mlstac
print("Loading dataset via mlstac.load()...")
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    ds = mlstac.load(snippet="isp-uv-es/CloudSEN12Plus")
print("Dataset loaded.")

# Apply filters on metadata: split 'test', label_type 'high', proj_shape == 509
print("Applying metadata filters...")
metadata_filtered = ds.metadata[
    (ds.metadata["split"] == "test") & 
    (ds.metadata["label_type"] == "high") & 
    (ds.metadata["proj_shape"] == 509)
]
print(f"Number of images after filtering: {len(metadata_filtered)}")

# Download datapoints
print("Downloading datapoints via mlstac.get_data()...")
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    datacube = mlstac.get_data(dataset=metadata_filtered)
print("Datapoints downloaded.")
print(f"Total datapoints downloaded: {len(datacube)}")

# Compute diversity (number of unique classes) per datapoint (using channel 13)
print("Computing label diversity for each datapoint...")
num_classes_list = []
for dp in tqdm(datacube, desc="Computing classes"):
    label = dp[13]
    unique_labels = np.unique(label)
    n_cls = len(unique_labels)
    if n_cls > 4:
        n_cls = 4
    num_classes_list.append(n_cls)

num_classes_arr = np.array(num_classes_list)
dist = {i: int(np.sum(num_classes_arr == i)) for i in [1, 2, 3, 4]}
print("Distribution of images by number of classes:", dist)

# Stratified selection for splits
print("Performing stratified selection for splits...")
N_total = len(datacube)
if N_total < TOTAL_REQUIRED:
    print(f"Warning: dataset has only {N_total} images, but {TOTAL_REQUIRED} are required.")
    sample_indices = list(range(N_total))
else:
    sample_indices = random.sample(range(N_total), TOTAL_REQUIRED)

sample_datapoints = [datacube[i] for i in sample_indices]
sample_strat = [num_classes_list[i] for i in sample_indices]

# First split: extract test split (NUM_TEST images)
train_val_dp, test_dp, train_val_strat, test_strat = train_test_split(
    sample_datapoints, sample_strat, 
    test_size=NUM_TEST / TOTAL_REQUIRED, 
    stratify=sample_strat, random_state=42
)
# Second split: from train_val, extract validation (NUM_VAL images) and train (remaining)
train_dp, val_dp, _, _ = train_test_split(
    train_val_dp, train_val_strat, 
    test_size=NUM_VAL / len(train_val_dp), 
    stratify=train_val_strat, random_state=42
)

print(f"Number of training images: {len(train_dp)}")
print(f"Number of validation images: {len(val_dp)}")
print(f"Number of test images: {len(test_dp)}")

# Save splits to disk
print("Saving splits to pickle files...")
with open("train_split.pkl", "wb") as f:
    pickle.dump(train_dp, f)
with open("val_split.pkl", "wb") as f:
    pickle.dump(val_dp, f)
with open("test_split.pkl", "wb") as f:
    pickle.dump(test_dp, f)
print("Splits saved.")

end_time = time.time()
print(f"Total time for dataset preparation: {end_time - start_time:.2f} seconds.")

# Visualize a sample from training split
print("Visualizing a sample from the training split...")
sample_dp = train_dp[0]
rgb = np.moveaxis(sample_dp[[3, 2, 1]], 0, -1) / 5000.0
human_label = sample_dp[13]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title("RGB Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(human_label, cmap="gray")
plt.title("Human Label")
plt.axis("off")
plt.tight_layout()
plt.show()
