# -*- coding: utf-8 -*-
from handle_datasets.old_load_subsets import *
import pandas as pd
from handle_datasets.old_paths import *

pd.set_option('display.expand_frame_repr', False, 'display.max_rows', None, 'display.max_rows', None)

"""
Inspect preprocessed datasets to make sure everything is correct
Creates a small sample CSV that can be inspected easily (10 rows)
Create folder inspect_sample in /data before running
"""


def inspect_fake_and_real():

    fake, real = assemble_datasets_after_split()

    print("Shape of fake: ", fake.shape)
    print("Shape of real:", real.shape)

    fake_sample = fake.head(10)
    real_sample = real.head(10)

    print(fake_sample.dtypes)
    print(real_sample.dtypes)

    print(fake_sample)
    print(real_sample)

    fake_sample.to_csv(INSPECT_FAKE_DATASET_PATH, encoding='utf-8')
    real_sample.to_csv(INSPECT_REAL_DATASET_PATH, encoding='utf-8')


def inspect_train_and_test():
    train = get_train_balanced_subset()
    test = get_test_subset()

    print("Shape of train: ", train.shape)
    print("Shape of test: ", test.shape)

    train_sample = train.head(10)
    test_sample = test.head(10)

    train_sample.to_csv(INSPECT_TRAIN_DATASET_PATH, encoding='utf-8')
    test_sample.to_csv(INSPECT_TEST_DATASET_PATH, encoding='utf-8')


inspect_fake_and_real()
