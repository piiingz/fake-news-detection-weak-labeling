import string

import pandas as pd

from config import NUMERICAL_COLS, TEXT_COLS
from handle_datasets.paths import DATASET_PATHS, SPLIT_PATHS, SPLIT_FILES_POSTFIXES, FILE_TYPES, TRAIN_BALANCED, TEST_BALANCED, \
    TRAIN_UNBALANCED, SPLIT_COMBINATIONS, TEST_SET_PATH_FAKE, TEST_SET_PATH_REAL, TEST_SET_PATH_FULL, TEST_SET_ALL_PREPROCESSED

pd.set_option('display.expand_frame_repr', False, 'display.max_rows', None)


""" Utils """


# Generic load CSV function
def load_dataframe_from_csv(path: string, include_columns: list = None, index_column=None):
    df = pd.read_csv(filepath_or_buffer=path, usecols=include_columns, index_col=index_column)

    return df


# Generic load Pickle function
def load_dataframe_from_pkl(path: string, include_columns: list = None):
    df = pd.read_pickle(path)

    if include_columns:
        df = df[include_columns]

    return df


# To check that everything is correct
def preview_pkl_split(dataset, split_class, split_number, nrows: int = 5):
    print(split_class, split_number)
    path = create_split_path(dataset, split_class, split_number, 'pkl')
    df = pd.read_pickle(path)
    print(path)

    preview = df.head(nrows)
    save_path = create_split_path(dataset, split_class, split_number, 'csv-preview')
    preview.to_csv(save_path, index=False)


def preview_pkl_full(dataset, nrows: int = 5):

    dataset_path = DATASET_PATHS[dataset] + '.pkl'

    df = pd.read_pickle(dataset_path)
    preview = pd.concat([df.head(nrows), df.tail(nrows)])

    save_path = DATASET_PATHS[dataset] + FILE_TYPES['csv-preview']
    preview.to_csv(save_path, index=False)


# Dataset: number from 2-4, split_class 1 = fake, 0 = real, split_number: 0 - 4
def create_split_path(dataset, split_class, split_number, file_type):
    dataset_path = DATASET_PATHS[dataset] + SPLIT_PATHS[split_class] + SPLIT_FILES_POSTFIXES[split_number] + FILE_TYPES[file_type]
    return dataset_path


""" Load datasets functions """


# Load dataset 1 - Full dataset with original features
def load_dataset_1():

    df = load_dataframe_from_csv(DATASET_PATHS[1], index_column=0)

    df.dropna(axis=0, subset=['title'], inplace=True)
    df.dropna(axis=0, subset=['content'], inplace=True)

    df = df.rename_axis('index_col').reset_index()

    return df


# Load dataset 2 - Original features in splits (from CSV)
def load_dataset_2_split(split_class, split_number):

    dataset_path = create_split_path(2, split_class, split_number, 'csv')

    df = load_dataframe_from_csv(dataset_path, index_column=0)

    df.dropna(axis=0, subset=['title'], inplace=True)
    df.dropna(axis=0, subset=['content'], inplace=True)

    df = df.rename_axis('index_col').reset_index()

    return df


# Load dataset 3 - All generated features except embeddings in splits (from PKL)
def load_dataset_3_split(split_class, split_number, include_columns: list = None):
    dataset_path = create_split_path(3, split_class, split_number, 'pkl')

    df = load_dataframe_from_pkl(dataset_path, include_columns)

    return df


def load_dataset_3_split_numerical(split_class, split_number):
    dataset_path = create_split_path(3, split_class, split_number, 'pkl')

    df = load_dataframe_from_pkl(dataset_path, include_columns=NUMERICAL_COLS)

    return df


def load_dataset_3_complete_text():

    df = pd.DataFrame()

    for split_combo in SPLIT_COMBINATIONS:
        split_class = split_combo[0]
        split_number = split_combo[1]

        split_df = load_dataset_3_split(split_class, split_number, include_columns=TEXT_COLS)
        df = df.append(split_df)

    return df


# Load dataset 4 - Word embeddings in splits (from PKL)
def load_dataset_4_split(split_class, split_number):
    dataset_path = create_split_path(4, split_class, split_number, 'pkl')

    df = load_dataframe_from_pkl(dataset_path)

    return df


# Load dataset 5 - Full dataset with embeddings (from PKL)
def load_dataset_5():
    df = load_dataframe_from_pkl(DATASET_PATHS[5] + '.pkl')

    return df


# Load dataset 6 - Full dataset with numerical, true and weak labels (from PKL)
def load_dataset_6_pkl():
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + '.pkl')

    return df


# Load dataset 6 - Full dataset with numerical, true and weak labels (from CSV)
def load_dataset_6_csv():
    df = load_dataframe_from_csv(DATASET_PATHS[6] + '.csv')

    return df


# Load numerical train set balanced
def load_dataset_6_train_balanced():
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + TRAIN_BALANCED + '.pkl')

    return df


# Load numerical test set balanced
def load_dataset_6_test_balanced():
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + TEST_BALANCED + '.pkl')

    return df[:1380]


# Load numerical train unbalanced
def load_dataset_6_train_unbalanced():
    df = load_dataframe_from_pkl(DATASET_PATHS[6] + TRAIN_UNBALANCED + '.pkl')

    return df


# Load full
def load_dataset_test_full():
    return load_dataframe_from_csv(TEST_SET_PATH_FULL)


# Load manually created preprocessed dataset
def load_dataset_test_preprocessed_pkl(include: list = None):
    return load_dataframe_from_pkl(TEST_SET_ALL_PREPROCESSED + ".pkl", include_columns=include)


# Load manually created preprocessed dataset
def load_dataset_test_preprocessed_csv(include: list = None):
    return load_dataframe_from_csv(TEST_SET_ALL_PREPROCESSED + ".csv", include_columns=include)


# Load fake and real 
def load_dataset_test_real():
    return load_dataframe_from_csv(TEST_SET_PATH_REAL, index_column=0)


def load_dataset_test_fake():
    return load_dataframe_from_csv(TEST_SET_PATH_FAKE, index_column=0)
