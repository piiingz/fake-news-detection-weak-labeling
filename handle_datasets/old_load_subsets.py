# -*- coding: utf-8 -*-
import pandas as pd
from handle_datasets.old_paths import *
from config import NUMERICAL_COLS, LIST_TYPE_COLS
from ast import literal_eval


def get_subsets():
    train_subset = pd.read_csv(TRAIN_READ_CSV_PATH, index_col=[0], encoding='utf-8')
    validation_subset = pd.read_csv(VALIDATION_READ_CSV_PATH, index_col=[0], encoding='utf-8')
    test_subset = pd.read_csv(TEST_READ_CSV_PATH, index_col=[0], encoding='utf-8')

    # Convert stored lists from string to list
    for col in LIST_TYPE_COLS:
        if col in train_subset.columns:
            train_subset[col] = train_subset[col].apply(literal_eval)
            validation_subset[col] = validation_subset[col].apply(literal_eval)
            test_subset[col] = test_subset[col].apply(literal_eval)

    return train_subset, validation_subset, test_subset


def get_test_original_features_subset():
    test_subset = pd.read_csv(TEST_READ_ORIGINAL_CSV_PATH, index_col=[0], encoding='utf-8')

    # Convert stored lists from string to list
    for col in LIST_TYPE_COLS:
        if col in test_subset.columns:
            test_subset[col] = test_subset[col].apply(literal_eval)

    return test_subset


def get_subsets_fake_and_real():
    fake_subset = pd.read_csv(FAKE_READ_CSV_PATH, index_col=[0], encoding='utf-8')
    real_subset = pd.read_csv(REAL_READ_CSV_PATH, index_col=[0], encoding='utf-8')

    print("Converting to lists")
    # Convert stored lists from string to list
    for col in LIST_TYPE_COLS:
        if col in fake_subset.columns:
            fake_subset[col] = fake_subset[col].apply(literal_eval)
            real_subset[col] = real_subset[col].apply(literal_eval)

    fake_subset.dropna(axis=0, subset=['title'], inplace=True)
    fake_subset.dropna(axis=0, subset=['content'], inplace=True)
    real_subset.dropna(axis=0, subset=['title'], inplace=True)
    real_subset.dropna(axis=0, subset=['content'], inplace=True)

    print("Done")

    return fake_subset, real_subset


def get_subset_split(path):
    subset = pd.read_csv(path, index_col=[0], encoding='utf-8')

    print("Converting to lists")
    # Convert stored lists from string to list
    for col in LIST_TYPE_COLS:
        if col in subset.columns:
            subset[col] = subset[col].apply(literal_eval)

    subset.dropna(axis=0, subset=['title'], inplace=True)
    subset.dropna(axis=0, subset=['content'], inplace=True)

    print("Done")

    return subset


def get_subset_split_features(path, features):
    subset = pd.read_csv(path, usecols=features, encoding='utf-8')

    print("Converting to lists")
    # Convert stored lists from string to list
    for col in LIST_TYPE_COLS:
        if col in subset.columns:
            subset[col] = subset[col].apply(literal_eval)

    for f in features:
        subset.dropna(axis=0, subset=[f], inplace=True)

    print("Done")

    return subset


def get_numerical_subset(path):
    subset = pd.read_csv(path, encoding='utf-8', usecols=NUMERICAL_COLS)

    for col in LIST_TYPE_COLS:
        if col in subset.columns:
            subset[col] = subset[col].apply(literal_eval)

    return subset


def get_pos_tagged_subset(path):
    subset = pd.read_csv(path, encoding='utf-8', usecols=['content_pos_tags', 'title_pos_tags'])
    print(subset.columns)
    for col in LIST_TYPE_COLS:
        if col in subset.columns:
            subset[col] = subset[col].apply(literal_eval)

    return subset

  
def get_train_total_subset():
    train_subset = pd.read_csv(TRAIN_TOTAL_PREPROCESSED_PATH, encoding='utf-8')

    for col in LIST_TYPE_COLS:
        if col in train_subset.columns:
            train_subset[col] = train_subset[col].apply(literal_eval)

    return train_subset


def get_train_balanced_subset(usecols: list = None):
    train_subset = pd.read_csv(TRAIN_BALANCED_PREPROCESSED_PATH, encoding='utf-8', usecols=usecols)

    for col in LIST_TYPE_COLS:
        if col in train_subset.columns:
            train_subset[col] = train_subset[col].apply(literal_eval)

    return train_subset


def get_test_subset(usecols: list = None):
    test_subset = pd.read_csv(TEST_PREPROCESSED_PATH, encoding='utf-8', usecols=usecols)

    for col in LIST_TYPE_COLS:
        if col in test_subset.columns:
            test_subset[col] = test_subset[col].apply(literal_eval)

    test_subset.dropna(axis=0, subset=['title'], inplace=True)
    test_subset.dropna(axis=0, subset=['content'], inplace=True)

    return test_subset


def assemble_datasets_after_split():
    # Assemble preprocessed dataset

    fake_total = pd.DataFrame()
    real_total = pd.DataFrame()

    for file_name in FAKE_FILE_POSTFIXES:

        load_path = FAKE_SAVE_PATH_PREFIX + file_name
        fake_subset = pd.read_csv(load_path, index_col=[0], encoding='utf-8')
        fake_total = fake_total.append(fake_subset)

    for file_name in REAL_FILE_POSTFIXES:

        load_path = REAL_SAVE_PATH_PREFIX + file_name
        real_subset = pd.read_csv(load_path, index_col=[0], encoding='utf-8')
        real_total = real_total.append(real_subset)

    return fake_total, real_total


def split_datasets():
    fake_subset = pd.read_csv(FAKE_READ_CSV_PATH, index_col=[0], encoding='utf-8')
    real_subset = pd.read_csv(REAL_READ_CSV_PATH, index_col=[0], encoding='utf-8')

    fake_rows = fake_subset.shape[0]
    real_rows = real_subset.shape[0]

    last_i = 0
    for i in range(0, fake_rows, 100000):
        fake_subset[i:i+100000].to_csv('data/all/fake_split/split_' + str(i) + '.csv', encoding='utf-8')
        last_i = i
    fake_subset[last_i:].to_csv('data/all/fake_split/split_' + str(last_i) + '.csv', encoding='utf-8')

    for i in range(0, real_rows, 100000):
        real_subset[i:i+100000].to_csv('data/all/real_split/split_' + str(i) + '.csv', encoding='utf-8')
        last_i = i
    real_subset[last_i:].to_csv('data/all/real_split/split_' + str(last_i) + '.csv', encoding='utf-8')


def convert_dataset():
    fake_subset = pd.read_csv(FAKE_READ_CSV_PATH, index_col=[0], encoding='utf-8')
    real_subset = pd.read_csv(REAL_READ_CSV_PATH, index_col=[0], encoding='utf-8')

    fake_subset.to_csv(FAKE_SAVE_TO_CSV_PATH, encoding='utf-8')
    real_subset.to_csv(REAL_SAVE_TO_CSV_PATH, encoding='utf-8')


def remove_index():
    train_subset, validation_subset, test_subset = get_subsets()

    # train_subset = train_subset.drop('Unnamed: 0.1', 1)
    # train_subset = train_subset.drop('id_y', 1)
    # train_subset = train_subset.rename(columns={'id_x': 'article_id'})
    # train_subset = train_subset.drop('id.1', 1)
    # train_subset.index.name = 'id'
    # train_subset.to_csv('../data/train_subset.csv')
    # validation_subset = validation_subset.drop('id_y', 1)
    # validation_subset = validation_subset.rename(columns={'id_x': 'article_id'})
    # validation_subset.to_csv('../data/validation_su

    # test_subset = test_subset.drop('id_y', 1)
    # test_subset = test_subset.rename(columns={'id_x': 'article_id'})
    # test_subset.to_csv('../data/test_subset.csv')

    print(train_subset.columns)
    print(validation_subset.columns)
    print(test_subset.columns)


def save_numerical_features_to_csv(subset: pd.DataFrame, path: str):
    numerical = subset.select_dtypes('number')
    print("Numerical shape: ", numerical.shape)
    numerical.to_csv(path, encoding='utf-8')
