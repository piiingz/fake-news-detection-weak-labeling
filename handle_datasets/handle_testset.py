import pandas as pd
from handle_datasets.load_datasets import load_dataset_test_fake, load_dataset_test_real, load_dataset_test_preprocessed_pkl, \
    load_dataset_test_full
from feature_engineering.run_all_preprocesing import run_all_preprocessing_dataframe
from handle_datasets.paths import TEST_SET_PATH_FULL
from handle_datasets.save_datasets import save_testset_all
from config import TEST_NUMERICAL_COLS, LEMMATIZED_COLS, RAW_TEXT_COLS

RANDOM_SEED = 2021


def combine_testset():
    df_fake = load_dataset_test_fake()
    df_real = load_dataset_test_real()

    df = pd.concat([df_fake, df_real]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(df)
    df.to_csv(TEST_SET_PATH_FULL)


def preprocess_testset():
    df = load_dataset_test_full()
    df = run_all_preprocessing_dataframe(df)
    save_testset_all(df)


def get_numerical_testset():
    return load_dataset_test_preprocessed_pkl(include=TEST_NUMERICAL_COLS)
    

def get_lemmatized_testset():
    num = get_numerical_testset()
    lem = load_dataset_test_preprocessed_pkl(include=LEMMATIZED_COLS)
    return pd.merge(num, lem, on='id')


def get_raw_testset():
    num = get_numerical_testset()
    raw = load_dataset_test_preprocessed_pkl(include=RAW_TEXT_COLS)
    return pd.merge(num, raw, on='id')
