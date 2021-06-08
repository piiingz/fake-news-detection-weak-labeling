# -*- coding: utf-8 -*-
from feature_engineering.run_all_preprocesing import run_all_preprocessing_splits
import nltk
from handle_datasets.old_paths import *

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('movie_reviews')
nltk.download('stopwords')

POSTFIXES = ['0.csv', '100000.csv', '0.csv', '100000.csv', '200000.csv', '300000.csv', '400000.csv']

index = 0


def run(path_prefix, save_path_prefix, postfix):
    print("Fetching subset..")
    path = path_prefix + postfix
    save_path = save_path_prefix + postfix
    run_all_preprocessing_splits(path, save_path)


run(FAKE_PATH_PREFIX, FAKE_SAVE_PATH_PREFIX, POSTFIXES[index])
