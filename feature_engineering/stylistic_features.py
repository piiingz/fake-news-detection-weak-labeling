# -*- coding: utf-8 -*-
import pandas as pd
from nltk.corpus import stopwords


def count_words(subset: pd.DataFrame):
    subset['content_word_count'] = subset['content_tokenized_words'].apply(lambda x: len(x))
    subset['title_word_count'] = subset['title_tokenized_words'].apply(lambda x: len(x))

    subset['content_word_count_with_punctuation'] = subset['content_tokenized_words_with_punctuation'].apply(lambda x: len(x))
    subset['title_word_count_with_punctuation'] = subset['title_tokenized_words_with_punctuation'].apply(lambda x: len(x))

    return subset


def count_sentences(subset: pd.DataFrame):
    subset['content_sentence_count'] = subset['content_tokenized_sentences'].apply(lambda x: len(x))
    subset['title_sentence_count'] = subset['title_tokenized_sentences'].apply(lambda x: len(x))
    return subset


def count_stop_words(subset: pd.DataFrame):

    def count(article):
        stop_words = stopwords.words('english')
        stop_word_count = 0

        for word in article:
            if word in stop_words:
                stop_word_count += 1

        if len(article):
            return pd.Series([stop_word_count, stop_word_count/len(article)])
        return pd.Series([stop_word_count, 0])

    subset[['content_stop_word_count', 'content_stop_word_ratio']] = subset['content_tokenized_words_lowercase'].apply(count)
    subset[['title_stop_word_count', 'title_stop_word_ratio']] = subset['title_tokenized_words_lowercase'].apply(count)
    return subset


def count_capital_words(subset: pd.DataFrame):

    def count_capital(words):

        uppercase_count = 0

        for word in words:
            if word.isupper() and len(word) > 1:
                uppercase_count += 1

        if len(words):
            return pd.Series([uppercase_count, uppercase_count / len(words)])
        return pd.Series([uppercase_count, 0])

    subset[['content_capital_word_count', 'content_capital_word_ratio']] = subset['content_tokenized_words'].apply(count_capital)
    subset[['title_capital_word_count', 'title_capital_word_ratio']] = subset['title_tokenized_words'].apply(count_capital)

    return subset


def count_urls(subset: pd.DataFrame):

    def count(article: str):
        counter = 0
        urls = []
        if "http" in article or "www" in article:
            words = article.split()
            for word in words:
                if "http" in word or "www" in word:
                    counter += 1
                    urls.append(word)
        return pd.Series([counter, urls])

    subset[['content_url_count', 'content_urls']] = subset['content'].apply(count)
    return subset


def count_quotes(subset: pd.DataFrame):
    quotes = '"“”«»'

    def count(article: list):
        quote_count = 0

        for word in article:
            if word in quotes:
                quote_count += 1

        if len(article):
            return pd.Series([quote_count, quote_count / len(article)])
        return pd.Series([quote_count, 0])

    subset[["content_quote_marks_count", "content_quote_marks_ratio"]] = subset[
        "content_tokenized_words_with_punctuation"].apply(count)

    return subset


def count_exclamation_marks(subset: pd.DataFrame):

    def exclamation_counter(text, sentence_count):
        if sentence_count:
            count = 0
            for char in text:
                if char == '!':
                    count += 1

            return pd.Series([count, count/sentence_count])
        return pd.Series([0, 0])

    subset[['content_exclamation_count', 'content_exclamation_ratio']] = subset.apply(
        lambda x: exclamation_counter(x.content, x.content_sentence_count), axis=1)
    subset[['title_exclamation_count', 'title_exclamation_ratio']] = subset.apply(
        lambda x: exclamation_counter(x.title, x.title_sentence_count), axis=1)
    return subset
