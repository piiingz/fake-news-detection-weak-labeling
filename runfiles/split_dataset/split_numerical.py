# -*- coding: utf-8 -*-

"""
1. Retrieve train dataset
2. Create descriptions
3. Create plots
"""
from handle_datasets.old_load_subsets import get_train_total_subset, save_numerical_features_to_csv
from handle_datasets.old_paths import TRAIN_TOTAL_NUMERICAL_COLS_PATH

train_subset = get_train_total_subset()
print("Train shape: ", train_subset.shape)
save_numerical_features_to_csv(train_subset, TRAIN_TOTAL_NUMERICAL_COLS_PATH)
