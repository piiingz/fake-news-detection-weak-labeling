from config import IDUN


if IDUN:
    DATA_PATH = 'data/'
else:
    DATA_PATH = '../data/'

NEW_FEATURES = DATA_PATH + 'new_features/'
ORIGINAL_FEATURES = DATA_PATH + 'original_features/'
ALL = DATA_PATH + 'all/'
ALL_PREPROCESSED = DATA_PATH + 'all_preprocessed/'
DESCRIBE = DATA_PATH + 'describe/'
WEAK_LABELING = DATA_PATH + 'weak_labeling/'
SNUBA_RESULTS  = DATA_PATH + 'snuba/results/'
SNUBA_GOAL = DATA_PATH + 'snuba/goals/'

NO1_ORIGINAL = DATA_PATH + 'no1_original/original.csv'
NO2_ORIGINAL_SPLIT = DATA_PATH + 'no2_original_split/'
NO3_ALL_FEATURES_SPLIT = DATA_PATH + 'no3_all_features_split/'
NO4_EMBEDDINGS_SPLIT = DATA_PATH + 'no4_embeddings_split/'
NO5_EMBEDDINGS = DATA_PATH + 'no5_embeddings/embeddings'
NO6_NUMERICAL = DATA_PATH + 'no6_numerical/numerical'
# NO7_NUMERICAL_SNUBA_LABELS = DATA_PATH + 'no7_numerical_snuba/labeled_numerical'

FAKE_SPLIT = 'fake_split/'
REAL_SPLIT = 'real_split/'

TRAIN_BALANCED = '_train_balanced'
TEST_BALANCED = '_test_balanced'
TRAIN_UNBALANCED = '_train_unbalanced'

TRAIN_BALANCED_LEMMA = '_train_balanced_lemma'
TEST_BALANCED_LEMMA = '_test_balanced_lemma'
TRAIN_UNBALANCED_LEMMA = '_train_unbalanced_lemma'

TRAIN_BALANCED_RAW = '_train_balanced_raw'
TEST_BALANCED_RAW = '_test_balanced_raw'
TRAIN_UNBALANCED_RAW = '_train_unbalanced_raw'


NO1_ORIGINAL_LOCAL_TEST = NO1_ORIGINAL + 'original.csv'

SPLIT_FILES_POSTFIXES = ['split_0', 'split_100000', 'split_200000', 'split_300000', 'split_400000', 'split_500000',
                         'split_600000', 'split_700000', 'split_800000', 'split_900000',  'split_1000000',
                         'split_1100000', 'split_1200000', 'split_1300000']

FILE_TYPES = {'csv': '.csv', 'pkl': '.pkl', 'csv-preview': '_preview.csv'}


""" Help dictionaries """
DATASET_PATHS = {1: NO1_ORIGINAL, 2: NO2_ORIGINAL_SPLIT, 3: NO3_ALL_FEATURES_SPLIT, 4: NO4_EMBEDDINGS_SPLIT,
                 5: NO5_EMBEDDINGS, 6: NO6_NUMERICAL}


SPLIT_PATHS = {1: FAKE_SPLIT, 0: REAL_SPLIT}

# 1st number is split path, 2nd number is split postfix index
SPLIT_COMBINATIONS = [(1, 0), (1, 1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

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


""" Labeling """
LF_SUMMARY_PATH = WEAK_LABELING + 'analysis/summary'
LF_LABEL_ANALYSIS_PATH = WEAK_LABELING + 'analysis/label_analysis'
LF_CONFUSION_MATRIX_PATH = WEAK_LABELING + 'confusion_matrix/conf_'
LF_EMPIRICAL_ACCURACIES_PATH = WEAK_LABELING + 'analysis/empirical_accuracies'

TRAIN_FILTERED_WEAK_LABELED_PATH = WEAK_LABELING + 'train_filtered'
TRAIN_BALANCED_WEAK_LABELED_PATH = WEAK_LABELING + 'train_balanced'
LABEL_MODEL_SCORE_PATH = WEAK_LABELING + 'scores'

""" Threshold generation """
THRESHOLD_PATH = WEAK_LABELING + 'thresholds_'

""" Snuba """
SNUBA_LABELS_UNLABELED_PATH = SNUBA_RESULTS + '1380_snuba_labels_unlabeled_' 
SNUBA_LABELS_LABELED_PATH = SNUBA_RESULTS + '1380_snuba_labels_labeled_' 
SNUBA_HEURISTICS_STATS_PATH = SNUBA_RESULTS + '1380_heuristic_stats_' 
SNUBA_HEURISTICS_PATH = SNUBA_RESULTS + '1380_heuristics_' 
SNUBA_GEN_MODEL_PATH = SNUBA_RESULTS + '1380_gen_model_' 
SNUBA_BETA_OPT_PATH = SNUBA_RESULTS + '1380_beta_opt_' 
SNUBA_FEAT_COMBOS_PATH = SNUBA_RESULTS + '1380_feat_combos_' 
SNUBA_SCORES_PATH = SNUBA_RESULTS + '1380_scores_'
SNUBA_SCORES_PATH_SNORKEL_GM = SNUBA_RESULTS + '1380_snorkel_scores_gm_'
SNUBA_SCORES_PATH_SNORKEL_MV = SNUBA_RESULTS + '1380_snorkel_scores_mv_'

""" Snuba test """
SNUBA_TEST_LABELS_UNLABELED_PATH = SNUBA_RESULTS + '1380_snuba_test_labels_test_unlabeled_' 
SNUBA_TEST_LABELS_LABELED_PATH = SNUBA_RESULTS + '1380_snuba_test_labels_labeled_' 
SNUBA_TEST_HEURISTICS_STATS_PATH = SNUBA_RESULTS + '1380_test_heuristic_stats_' 
SNUBA_TEST_HEURISTICS_PATH = SNUBA_RESULTS + '1380_test_heuristics_' 
SNUBA_TEST_GEN_MODEL_PATH = SNUBA_RESULTS + '1380_test_gen_model_' 
SNUBA_TEST_BETA_OPT_PATH = SNUBA_RESULTS + '1380_test_beta_opt_' 
SNUBA_TEST_FEAT_COMBOS_PATH = SNUBA_RESULTS + '1380_test_feat_combos_' 
SNUBA_TEST_SCORES_PATH = SNUBA_RESULTS + '1380_test_scores_'
SNUBA_TEST_SCORES_PATH_SNORKEL_GM = SNUBA_RESULTS + '1380_test_snorkel_scores_gm_'
SNUBA_TEST_SCORES_PATH_SNORKEL_MV = SNUBA_RESULTS + '1380_test_snorkel_scores_mv_'


""" Snuba apply heuristics """
SNUBA_GOAL_HEURISTICS_PATH = SNUBA_GOAL + 'heuristics.csv'
SNUBA_GOAL_GEN_MODEL_PATH = SNUBA_GOAL + 'gen_model.csv'
SNUBA_GOAL_BETA_OPT_PATH = SNUBA_GOAL + 'beta_opt.csv'
SNUBA_GOAL_FEAT_COMBOS_PATH = SNUBA_GOAL + 'feat_combos.csv'

""" Testset """
TEST_SET_PATH = DATA_PATH + "testset/"
TEST_SET_PATH_REAL = TEST_SET_PATH + "real_cleaned.csv"
TEST_SET_PATH_FAKE = TEST_SET_PATH + "fake_cleaned.csv"
TEST_SET_PATH_FULL = TEST_SET_PATH + "full.csv"
TEST_SET_ALL_PREPROCESSED = TEST_SET_PATH + "all_preprocessed"
TEST_SET_NUMERICAL = TEST_SET_PATH + "numerical"
TEST_SET_LEMMATIZED = TEST_SET_PATH + "lemmatized"
TEST_SET_RAW = TEST_SET_PATH + "raw"