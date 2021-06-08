import sys

import pandas as pd

from config import SPLIT_SIZE
from feature_engineering.glove_features import apply_glove_embedding
from feature_engineering.run_all_preprocesing import run_all_preprocessing_dataframe
from handle_datasets.load_datasets import load_dataset_2_split, load_dataset_1, preview_pkl_split, load_dataset_3_split, \
    load_dataset_4_split, preview_pkl_full, load_dataset_3_split_numerical
from handle_datasets.save_datasets import save_dataset_3_split, save_dataset_4_split, save_dataset_5, \
    save_dataset_6_pkl, save_dataset_6_csv
from handle_datasets.paths import SPLIT_FILES_POSTFIXES, SPLIT_COMBINATIONS, DATASET_PATHS, FILE_TYPES

dataset = sys.argv[1]    # = var1 = number between 2 - 6 (which dataset to generate)
split_combination = sys.argv[2] if len(sys.argv) > 2 else None   # var2 = index of split combination 0 - 6(see paths.py)
preview = sys.argv[3] if len(sys.argv) > 3 else False  # If you want to create a preview
# To run write in terminal/SLURM: python creata_datasets.py var1 var2


def create_dataset_2():
    df = load_dataset_1()
    save_splits_to_csv(DATASET_PATHS[2], df)


def create_dataset_3_split():
    df = load_dataset_2_split(split_class, split_number)
    df = run_all_preprocessing_dataframe(df)

    save_dataset_3_split(df, split_class, split_number)


# Needs to run after dataset 3
def create_dataset_4_split():
    df = load_dataset_3_split(split_class, split_number)
    df = apply_glove_embedding(df)

    df = df[['index_col', 'title_glove', 'content_glove']]

    save_dataset_4_split(df, split_class, split_number)


# Needs to run after dataset 4
def create_dataset_5():

    df = pd.DataFrame()

    # Fetch splits from dataset 4 and append
    for split_combo in SPLIT_COMBINATIONS:
        split_class = split_combo[0]
        split_number = split_combo[1]

        split_df = load_dataset_4_split(split_class, split_number)
        df = df.append(split_df)

    save_dataset_5(df)


# Numerical with labels
def create_dataset_6():

    df = pd.DataFrame()

    # Fetch numerical splits from dataset 3, add label and append
    for split_combo in SPLIT_COMBINATIONS:
        split_class = split_combo[0]
        split_number = split_combo[1]

        split_df = load_dataset_3_split_numerical(split_class, split_number)
        split_df['label'] = split_class
        df = df.append(split_df)

    save_dataset_6_pkl(df)
    save_dataset_6_csv(df)


def preview_split():
    preview_pkl_split(int(dataset), split_class, split_number)


def preview_full():
    preview_pkl_full(int(dataset))


def save_splits_to_csv(path, df):
    df_rows = df.shape[0]

    postfix = 0
    for i in range(0, df_rows, SPLIT_SIZE):
        df[i:i + SPLIT_SIZE].to_csv(path + SPLIT_FILES_POSTFIXES[postfix] + FILE_TYPES['csv'], encoding='utf-8')
        postfix += 1


if __name__ == '__main__':

    if split_combination is not None:
        split = SPLIT_COMBINATIONS[int(split_combination)]
        split_class = split[0]
        split_number = split[1]

    if preview:
        print('PREVIEW')

        preview_split() if int(preview) == 1 else preview_full()

    else:
        functions = {'2': create_dataset_2, '3': create_dataset_3_split, '4': create_dataset_4_split,
                     '5': create_dataset_5, '6': create_dataset_6}
        functions[dataset]()


