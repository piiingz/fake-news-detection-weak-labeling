# -*- coding: utf-8 -*-
from feature_engineering.run_all_preprocesing import run_all_preprocessing_splits
import nltk
from handle_datasets.old_paths import *

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('movie_reviews')
nltk.download('stopwords')


def run(path_prefix, save_path_prefix, file_postfixes):
    print("We're starting!")
    for postfix in file_postfixes:
        print("Fetching subset..")
        path = path_prefix + postfix
        save_path = save_path_prefix + postfix
        run_all_preprocessing_splits(path, save_path)


if __name__ == '__main__':
    run(FAKE_PATH_PREFIX, FAKE_SAVE_PATH_PREFIX, FAKE_FILE_POSTFIXES)
    run(REAL_PATH_PREFIX, REAL_SAVE_PATH_PREFIX, REAL_FILE_POSTFIXES)
