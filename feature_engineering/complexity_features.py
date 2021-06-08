# -*- coding: utf-8 -*-
import nltk
import pandas as pd


def ttr_score(subset: pd.DataFrame):
    # Type-token ratio

    def ttr_count(article):
        unique_word_count = nltk.Counter(article)

        if len(article):
            return len(unique_word_count) / len(article)
        return 0

    subset["content_ttr_score"] = subset["content_tokenized_words_lowercase"].apply(ttr_count)
    subset["title_ttr_score"] = subset["title_tokenized_words_lowercase"].apply(ttr_count)

    return subset


def words_per_sentence(subset: pd.DataFrame):
    subset['content_words_per_sentence'] = subset.apply(lambda x:
                                                        (x['content_word_count'] / x['content_sentence_count'])
                                                        if x['content_sentence_count'] != 0 else 0, axis=1)
    return subset


def average_word_length(subset: pd.DataFrame):
    def count_chars_per_word(words: list):
        if len(words):
            return len(''.join(words))/len(words)
        return 0

    subset['content_avg_word_length'] = subset['content_tokenized_words'].apply(count_chars_per_word)
    subset['title_avg_word_length'] = subset['title_tokenized_words'].apply(count_chars_per_word)
    subset['content_avg_word_length_no_stop_words'] = subset['content_words_no_stopwords'].apply(
        count_chars_per_word)
    subset['title_avg_word_length_no_stop_words'] = subset['title_words_no_stopwords'].apply(
        count_chars_per_word)
    return subset
