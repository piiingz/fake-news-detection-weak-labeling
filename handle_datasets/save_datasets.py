import string

import pandas as pd

from handle_datasets.load_datasets import create_split_path
from handle_datasets.paths import *

pd.set_option('display.expand_frame_repr', False, 'display.max_rows', None)


""" Utils"""


# Generic save CSV function
def save_dataframe_to_csv(df: pd.DataFrame, path: string):
    df.to_csv(path, index=False)


# Generic save Pickle function
def save_dataframe_to_pkl(df, path: string):
    df.to_pickle(path)


""" Save datasets"""


# Save dataset 3 - All generated features except embeddings in splits (To PKL)
def save_dataset_3_split(df: pd.DataFrame, split_class, split_number):

    dataset_path = create_split_path(3, split_class, split_number, 'pkl')
    save_dataframe_to_pkl(df, dataset_path)


# Save dataset 4 - Word embeddings in splits (To PKL)
def save_dataset_4_split(df: pd.DataFrame, split_class, split_number):

    dataset_path = create_split_path(4, split_class, split_number, 'pkl')
    save_dataframe_to_pkl(df, dataset_path)


# save dataset 5 - Full dataset with embeddings (To PKL)
def save_dataset_5(df: pd.DataFrame):
    save_dataframe_to_pkl(df, DATASET_PATHS[5] + '.pkl')


# save dataset 6 - Full dataset with numerical, true and weak labels (To PKL)
def save_dataset_6_pkl(df: pd.DataFrame):
    save_dataframe_to_pkl(df, DATASET_PATHS[6] + '.pkl')


# save dataset 6 - Full dataset with numerical, true and weak labels (To CSV)
def save_dataset_6_csv(df: pd.DataFrame):
    save_dataframe_to_csv(df, DATASET_PATHS[6] + '.csv')


# save both pkl and csv
def save_dataset_6(df: pd.DataFrame):
    save_dataset_6_pkl(df)
    save_dataset_6_csv(df)


# Save test and train
def save_numerical_test_train(train_balanced: pd.DataFrame, test_balanced: pd.DataFrame, train_unbalanced: pd.DataFrame):

    save_dataframe_to_pkl(train_balanced, DATASET_PATHS[6] + TRAIN_BALANCED + '.pkl')
    save_dataframe_to_csv(train_balanced, DATASET_PATHS[6] + TRAIN_BALANCED + '.csv')

    save_dataframe_to_pkl(test_balanced, DATASET_PATHS[6] + TEST_BALANCED + '.pkl')
    save_dataframe_to_csv(test_balanced, DATASET_PATHS[6] + TEST_BALANCED + '.csv')

    save_dataframe_to_pkl(train_unbalanced, DATASET_PATHS[6] + TRAIN_UNBALANCED + '.pkl')
    save_dataframe_to_csv(train_unbalanced, DATASET_PATHS[6] + TRAIN_UNBALANCED + '.csv')


# Save test and train numerical + lemmatized
def save_numerical_lemma_test_train(train_balanced_lemma: pd.DataFrame, test_balanced_lemma: pd.DataFrame, train_unbalanced_lemma: pd.DataFrame):

    save_dataframe_to_pkl(train_balanced_lemma, DATASET_PATHS[6] + TRAIN_BALANCED_LEMMA + '.pkl')
    save_dataframe_to_csv(train_balanced_lemma, DATASET_PATHS[6] + TRAIN_BALANCED_LEMMA + '.csv')

    save_dataframe_to_pkl(test_balanced_lemma, DATASET_PATHS[6] + TEST_BALANCED_LEMMA + '.pkl')
    save_dataframe_to_csv(test_balanced_lemma, DATASET_PATHS[6] + TEST_BALANCED_LEMMA + '.csv')

    save_dataframe_to_pkl(train_unbalanced_lemma, DATASET_PATHS[6] + TRAIN_UNBALANCED_LEMMA + '.pkl')
    save_dataframe_to_csv(train_unbalanced_lemma, DATASET_PATHS[6] + TRAIN_UNBALANCED_LEMMA + '.csv')


# Save test and train numerical + raw
def save_numerical_raw_test_train(train_balanced_raw: pd.DataFrame, test_balanced_raw: pd.DataFrame, train_unbalanced_raw: pd.DataFrame):

    save_dataframe_to_pkl(train_balanced_raw, DATASET_PATHS[6] + TRAIN_BALANCED_RAW + '.pkl')
    save_dataframe_to_csv(train_balanced_raw, DATASET_PATHS[6] + TRAIN_BALANCED_RAW + '.csv')

    save_dataframe_to_pkl(test_balanced_raw, DATASET_PATHS[6] + TEST_BALANCED_RAW + '.pkl')
    save_dataframe_to_csv(test_balanced_raw, DATASET_PATHS[6] + TEST_BALANCED_RAW + '.csv')

    save_dataframe_to_pkl(train_unbalanced_raw, DATASET_PATHS[6] + TRAIN_UNBALANCED_RAW + '.pkl')
    save_dataframe_to_csv(train_unbalanced_raw, DATASET_PATHS[6] + TRAIN_UNBALANCED_RAW + '.csv')


def save_testset_all(df: pd.DataFrame):
    save_dataframe_to_pkl(df, TEST_SET_ALL_PREPROCESSED + ".pkl")
    save_dataframe_to_csv(df, TEST_SET_ALL_PREPROCESSED + ".csv")


def save_testset_numerical(df: pd.DataFrame):
    save_dataframe_to_pkl(df, TEST_SET_NUMERICAL + ".pkl")
    save_dataframe_to_csv(df, TEST_SET_NUMERICAL + ".csv")


def save_testset_lemmatized(df: pd.DataFrame):
    save_dataframe_to_pkl(df, TEST_SET_LEMMATIZED + ".pkl")
    save_dataframe_to_csv(df, TEST_SET_LEMMATIZED + ".csv")


def save_testset_raw(df: pd.DataFrame):
    save_dataframe_to_pkl(df, TEST_SET_RAW + ".pkl")
    save_dataframe_to_csv(df, TEST_SET_RAW + ".csv")
    