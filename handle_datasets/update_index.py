import sys

import pandas as pd

from handle_datasets.load_datasets import load_dataset_3_split, load_dataset_4_split, load_dataset_6_pkl
from handle_datasets.save_datasets import save_dataset_3_split, save_dataset_4_split, save_dataset_6
from handle_datasets.paths import SPLIT_COMBINATIONS

dataset = sys.argv[1]   # which dataset to update indices
is_split = int(sys.argv[2])   # if the dataset is in splits (1=true or 0=false)


def update_index(load_function, save_function):

    if is_split:
        # index starts at 0
        i = 0

        for split_combo in SPLIT_COMBINATIONS:
            split_class = split_combo[0]
            split_number = split_combo[1]

            split_df = load_function(split_class, split_number)

            split_df['id'] = pd.RangeIndex(i, i + split_df.shape[0], 1)

            i = i + split_df.shape[0]

            print("Split:", split_class, split_number)
            print("Length of df:", split_df.shape[0])
            print("indices from-to:", split_df['id'].head(1).values[0], split_df['id'].tail(1).values[0])

            split_df = split_df.drop(['index_col'], axis=1, errors='ignore')

            save_function(split_df, split_class, split_number)

    else:
        # Only for full datasets
        df = load_function()

        df['id'] = pd.RangeIndex(0, df.shape[0], 1)

        print("Length:", df.shape[0])
        print("indices from-to:", df['id'].head(1).values[0], df['id'].tail(1).values[0])

        df = df.drop(['index_col'], axis=1, errors='ignore')

        save_function(df)


if __name__ == '__main__':
    load_functions = {'3': load_dataset_3_split, '4': load_dataset_4_split, '6': load_dataset_6_pkl}
    save_functions = {'3': save_dataset_3_split, '4': save_dataset_4_split, '6': save_dataset_6}

    load_function = load_functions[dataset]
    save_function = save_functions[dataset]

    update_index(load_function, save_function)
