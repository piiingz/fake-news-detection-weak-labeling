# -*- coding: utf-8 -*-

from feature_engineering.complexity_features import *
from feature_engineering.pos_features import *
from feature_engineering.stylistic_features import *
from feature_engineering.sentiwordnet_features import apply_swn_score
from feature_engineering.sentiment_features import *
from feature_engineering.tokenized_features import *
from handle_datasets.old_load_subsets import get_subsets, get_test_original_features_subset, get_subsets_fake_and_real, get_subset_split
from handle_datasets.old_paths import *

pd.set_option('display.expand_frame_repr', False)

# Each group is dependent on feature from earlier group
group_1 = [
    word_tokenize,
    sentence_tokenize,
    remove_punctuation,
    count_urls
]

group_2 = [
    remove_stop_words,
    lowercase,
    pos_tagging,
    lemmatize_tokens,
    count_words,
    count_sentences,
    count_capital_words,
    count_exclamation_marks,
]

group_3 = [
    title_count_pos,
    count_personal_pronouns,
    count_stop_words,
    words_per_sentence,
    count_quotes,
    ttr_score
]

group_4 = [
    title_count_nouns,
    title_count_proper_nouns,
    average_word_length,
    count_verbs_and_adjectives,
    word_sentiments,
    sentence_sentiments,
    text_sentiments,
    apply_swn_score
]

group_order = [group_1, group_2, group_3, group_4]


def load_from_csv():
    train_subset, validation_subset, test_subset = get_subsets()
    return train_subset, validation_subset, test_subset


def save_to_csv(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
    train.to_csv(TRAIN_SAVE_TO_CSV_PATH, encoding='utf-8')
    validation.to_csv(VALIDATION_SAVE_TO_CSV_PATH, encoding='utf-8')
    test.to_csv(TEST_SAVE_TO_CSV_PATH, encoding='utf-8')


def run_all_preprocessing():
    # Run when you need to update the whole dataset
    # Only loads dataset once, applies all functions, then saves at the end

    train_subset, validation_subset, test_subset = load_from_csv()

    group_nr = 1
    for group in group_order:
        print("Starting group ", group_nr)
        func_nr = 1
        for func in group:
            train_subset = func(train_subset)
            validation_subset = func(validation_subset)
            test_subset = func(test_subset)
            print("Finished function {}/{}".format(func_nr, len(group)))
            func_nr += 1
        print("Finished group ", group_nr)
        group_nr += 1

    save_to_csv(train_subset, validation_subset, test_subset)


def run_all_preprocessing_test():
    # To test locally

    test_subset = get_test_original_features_subset()

    group_nr = 1
    for group in group_order:
        print("Starting group ", group_nr)
        func_nr = 1
        for func in group:
            test_subset = func(test_subset)
            print("Finished function {}/{}".format(func_nr, len(group)))
            func_nr += 1
        print("Finished group ", group_nr)
        group_nr += 1

    print(test_subset.columns)
    print(test_subset.info())

    test_subset.to_csv(TEST_SAVE_TO_CSV_PATH, encoding='utf-8')


def run_all_preprocessing_fake_and_real():
    # Run when you need to update the whole dataset
    # Only loads dataset once, applies all functions, then saves at the end

    print("Fetching subsets..")
    fake_subset, real_subset = get_subsets_fake_and_real()
    print("Subsets fetched!")

    group_nr = 1
    for group in group_order:
        print("Starting group ", group_nr)
        func_nr = 1
        for func in group:
            fake_subset = func(fake_subset)
            real_subset = func(real_subset)
            print("Finished function {}/{}".format(func_nr, len(group)))
            func_nr += 1
        print("Finished group ", group_nr)
        group_nr += 1

    fake_subset.to_csv(FAKE_SAVE_TO_CSV_PATH, encoding='utf-8')
    real_subset.to_csv(REAL_SAVE_TO_CSV_PATH, encoding='utf-8')


def run_all_preprocessing_splits(path, save_path):
    # Run when you need to update the whole dataset
    # Only loads dataset once, applies all functions, then saves at the end

    subset = get_subset_split(path)
    print("Subset fetched!")

    group_nr = 1
    for group in group_order:
        print("Starting group ", group_nr)
        func_nr = 1
        for func in group:
            subset = func(subset)
            print("Finished function {}/{}".format(func_nr, len(group)))
            func_nr += 1
        print("Finished group ", group_nr)
        group_nr += 1

    subset.to_csv(save_path, encoding='utf-8')


def run_all_preprocessing_dataframe(df: pd.DataFrame):

    group_nr = 1
    for group in group_order:
        print("Starting group ", group_nr)
        func_nr = 1
        for func in group:
            df = func(df)
            print("Finished function {}/{}".format(func_nr, len(group)))
            func_nr += 1
        print("Finished group ", group_nr)
        group_nr += 1

    return df

