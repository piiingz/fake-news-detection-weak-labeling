# -*- coding: utf-8 -*-
from config import IDUN

""" Folders """


if IDUN:
    DATA_PATH = 'data/'
    EMBEDDINGS_PATH = 'pretrained_embeddings/'
else:
    DATA_PATH = '../data/'
    EMBEDDINGS_PATH = '../pretrained_embeddings/'

NEW_FEATURES = DATA_PATH + 'new_features/'
ORIGINAL_FEATURES = DATA_PATH + 'original_features/'
ALL = DATA_PATH + 'all/'
ALL_PREPROCESSED = DATA_PATH + 'all_preprocessed/'
DESCRIBE = DATA_PATH + 'describe/'
WEAK_LABELING = DATA_PATH + 'weak_labeling/'

""" Load from CSV """
TRAIN_READ_CSV_PATH = NEW_FEATURES + 'train_subset.csv'
VALIDATION_READ_CSV_PATH = NEW_FEATURES + 'validation_subset.csv'
TEST_READ_CSV_PATH = NEW_FEATURES + 'original.csv'

TEST_READ_ORIGINAL_CSV_PATH = ORIGINAL_FEATURES + 'original.csv'

FAKE_READ_CSV_PATH = ALL + 'fake_articles.csv'
REAL_READ_CSV_PATH = ALL + 'real_articles.csv'


""" Save to CSV """
TRAIN_SAVE_TO_CSV_PATH = NEW_FEATURES + 'train_subset.csv'
VALIDATION_SAVE_TO_CSV_PATH = NEW_FEATURES + 'validation_subset.csv'
TEST_SAVE_TO_CSV_PATH = NEW_FEATURES + 'original.csv'

FAKE_SAVE_TO_CSV_PATH = ALL_PREPROCESSED + 'fake_articles.csv'
REAL_SAVE_TO_CSV_PATH = ALL_PREPROCESSED + 'real_articles.csv'


""" Describe data """
FAKE_SUBSET_DESCRIBE_PATH = DESCRIBE + 'fake_subset_description.csv'
REAL_SUBSET_DESCRIBE_PATH = DESCRIBE + 'real_subset_description.csv'


""" Make box plots """
SUBSET_BOXPLOT_PATH = DESCRIBE + 'subset_box_plots'
BOXPLOT_PATH = DESCRIBE + 'box_plots'
SUBSET_BOXPLOT_FLIERS_PATH = DESCRIBE + 'subset_box_plots_fliers'


""" Make histograms """
SUBSET_HISTOGRAM_PATH = DESCRIBE + 'subset_histogram'
SUBSET_HISTOGRAM_FLIERS_PATH = DESCRIBE + 'subset_histogram_fliers'


""" Split data into files"""
FAKE_FILE_POSTFIXES = ['0.csv', '100000.csv']
REAL_FILE_POSTFIXES = ['0.csv', '100000.csv', '200000.csv', '300000.csv', '400000.csv']

FAKE_PATH_PREFIX = ALL + 'fake_split/split_'
FAKE_SAVE_PATH_PREFIX = ALL_PREPROCESSED + 'fake_split/split_'

REAL_PATH_PREFIX = ALL + 'real_split/split_'
REAL_SAVE_PATH_PREFIX = ALL_PREPROCESSED + 'real_split/split_'


""" Save test and train balanced and unbalanced (total)"""
TRAIN_BALANCED_PREPROCESSED_PATH = ALL_PREPROCESSED + 'train/train_balanced.csv'
TEST_PREPROCESSED_PATH = ALL_PREPROCESSED + 'test/test.csv'
TRAIN_TOTAL_PREPROCESSED_PATH = ALL_PREPROCESSED + 'train/train_total.csv'

TRAIN_TOTAL_NUMERICAL_COLS_PATH = ALL_PREPROCESSED + 'train/train_total_numerical.csv'
TRAIN_BALANCED_NUMERICAL_COLS_PATH = ALL_PREPROCESSED + 'train/train_balanced_numerical.csv'
TEST_NUMERICAL_COLS_PATH = ALL_PREPROCESSED + 'test/test_numerical.csv'


""" Save samples of preprocessed datasets to inspect"""
INSPECT_FAKE_DATASET_PATH = DATA_PATH + 'inspect_sample/fake_sample.csv'
INSPECT_REAL_DATASET_PATH = DATA_PATH + 'inspect_sample/real_sample.csv'
INSPECT_TRAIN_DATASET_PATH = DATA_PATH + 'inspect_sample/train_sample.csv'
INSPECT_TEST_DATASET_PATH = DATA_PATH + 'inspect_sample/test_sample.csv'


""" Labeling """
LF_SUMMARY_PATH = WEAK_LABELING + 'analysis/summary'
LF_LABEL_ANALYSIS_PATH = WEAK_LABELING + 'analysis/label_analysis'
LF_CONFUSION_MATRIX_PATH = WEAK_LABELING + 'confusion_matrix/conf_'
LF_EMPIRICAL_ACCURACIES_PATH = WEAK_LABELING + 'analysis/empirical_accuracies'

TRAIN_FILTERED_WEAK_LABELED_PATH = WEAK_LABELING + 'train_filtered'
TRAIN_BALANCED_WEAK_LABELED_PATH = WEAK_LABELING + 'train_balanced'
LABEL_MODEL_SCORE_PATH = WEAK_LABELING + 'scores'


""" Embeddings """
FAKE_EMBEDDINGS_SAVE_PATH = ALL_PREPROCESSED + 'embeddings/fake_split/split_'
REAL_EMBEDDINGS_SAVE_PATH = ALL_PREPROCESSED + 'embeddings/real_split/split_'

PRE_TRAINED_EMBEDDINGS = EMBEDDINGS_PATH + 'glove.6B.100d.txt.word2vec'
PRE_TRAINED_EMBEDDINGS_BEFORE_CONVERSION = EMBEDDINGS_PATH + 'glove.6B.100d.txt'


""" Threshold generation """
THRESHOLD_PATH = WEAK_LABELING + 'thresholds.csv'
