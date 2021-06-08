# -*- coding: utf-8 -*-

from handle_datasets.old_load_subsets import get_subsets
from describe.make_plots import *

""" Download necessary nltk corpora """
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('movie_reviews')
# nltk.download('stopwords')

pd.set_option('display.expand_frame_repr', False, 'display.max_rows', None)

train_subset, validation_subset, test_subset = get_subsets()


def apply_and_save_all_subsets(fun: callable):
    train_features = fun(train_subset)
    validation_features = fun(validation_subset)
    test_features = fun(test_subset)

    train_features.to_csv('../data/new_features/train_subset.csv', encoding='utf-8')
    validation_features.to_csv('../data/new_features/validation_subset.csv', encoding='utf-8')
    test_features.to_csv('../data/new_features/test_subset.csv', encoding='utf-8')


def test_feature(fun: callable):
    test_features = fun(train_subset)
    #print(test_features.head())


def print_columns():

    train_cols = train_subset.columns
    validation_cols = validation_subset.columns
    test_cols = test_subset.columns

    print(train_subset.dtypes)

    print(train_cols == validation_cols)
    print(validation_cols == test_cols)


def remove_rows(subset: pd.DataFrame, number):
    return subset.iloc[0: number, :]


def remove_column(subset: pd.DataFrame):

    drop_column = "urls_in_content"

    subset = subset.drop(drop_column, 1)
    return subset


if __name__ == "__main__":

    """ 1. Preprocessing """

    # apply_and_save_all_subsets(word_tokenize)
    # apply_and_save_all_subsets(sentence_tokenize)
    # apply_and_save_all_subsets(remove_punctuation)
    # apply_and_save_all_subsets(ngrams_tokenize)
    # apply_and_save_all_subsets(lowercase)
    # apply_and_save_all_subsets(remove_stop_words)
    # apply_and_save_all_subsets(pos_tagging)
    # apply_and_save_all_subsets(negate_sentences)

    """ Count features """

    # apply_and_save_all_subsets(count_exclamation_marks)
    # apply_and_save_all_subsets(count_words)
    # apply_and_save_all_subsets(count_sentences)
    # apply_and_save_all_subsets(count_capital_words)
    # apply_and_save_all_subsets(title_count_pos)
    # apply_and_save_all_subsets(count_personal_pronouns)
    # apply_and_save_all_subsets(count_stop_words)
    # apply_and_save_all_subsets(words_per_sentence)
    # apply_and_save_all_subsets(count_quotes)
    # apply_and_save_all_subsets(ttr_score)
    # apply_and_save_all_subsets(title_count_nouns)
    # apply_and_save_all_subsets(title_count_proper_nouns)
    # apply_and_save_all_subsets(average_word_length)
    # apply_and_save_all_subsets(count_urls)
    # apply_and_save_all_subsets(count_verbs_and_adjectives)

    # """ Sentiment features """
    # apply_and_save_all_subsets(word_sentiments)
    # apply_and_save_all_subsets(sentence_sentiments)
    # apply_and_save_all_subsets(text_sentiments)

    """ Other """
    # test_feature(count_exclamation_marks)
    # print_columns()
    # test_feature(count_verbs_and_adjectives)
    # apply_and_save_all_subsets(remove_column)
    # apply_and_save_all_subsets(remove_column)

    pass
