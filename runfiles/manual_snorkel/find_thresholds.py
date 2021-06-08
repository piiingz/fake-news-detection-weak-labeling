# -*- coding: utf-8 -*-

"""
1. Retrieve train dataset
2. Create descriptions
3. Create plots
"""

from handle_datasets.load_datasets import load_dataset_6_test_balanced
from describe.describe_features import describe_train_features
from describe.make_plots import subset_make_box_plots, subset_histogram

subset = load_dataset_6_test_balanced()
describe_train_features(subset)
subset_make_box_plots(subset)
subset_histogram(subset)

