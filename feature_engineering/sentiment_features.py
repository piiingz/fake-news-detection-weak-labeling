# -*- coding: utf-8 -*-

import pandas as pd
from nltk.sentiment.util import mark_negation
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

nb_analyzer = NaiveBayesAnalyzer()


""" Calculate sentiment functions """


def get_sentiment_from_tokens(tokens: list):
    count = len(tokens)
    total_sub = 0
    total_pos = 0
    total_neg = 0
    for token in tokens:
        blob_pattern = TextBlob(token)
        blob_nb = TextBlob(token, analyzer=nb_analyzer)
        total_sub += blob_pattern.sentiment.subjectivity
        total_pos += blob_nb.sentiment.p_pos
        total_neg += blob_nb.sentiment.p_neg
    if count:
        return pd.Series([total_sub / count, total_pos / count, total_neg / count])
    return pd.Series([0, 0, 0])


def get_sentiment_from_text(text: str):
    blob_pattern = TextBlob(text)
    blob_nb = TextBlob(text, analyzer=nb_analyzer)
    subjectivity = blob_pattern.sentiment.subjectivity
    pos, neg = blob_nb.sentiment.p_pos, blob_nb.sentiment.p_neg
    return pd.Series([subjectivity, pos, neg])


""" Alter data functions """


def negate_sentences(subset: pd.DataFrame):
    def negate(sentence: str):
        return mark_negation(sentence.split())

    subset['content_tokenized_sentences_negated'] = subset['content_tokenized_sentences_lowercase'].apply(
        lambda x: list(map(negate, x)))
    subset['title_tokenized_sentences_negated'] = subset['title_tokenized_sentences_lowercase'].apply(
        lambda x: list(map(negate, x)))
    return subset


def word_sentiments(subset: pd.DataFrame):
    subset[['content_sentiment_word_sub', 'content_sentiment_word_pos', 'content_sentiment_word_neg']] = subset[
        'content_words_no_stopwords'].apply(get_sentiment_from_tokens)
    subset[['title_sentiment_word_sub', 'title_sentiment_word_pos', 'title_sentiment_word_neg']] = subset[
        'title_words_no_stopwords'].apply(get_sentiment_from_tokens)
    return subset


def sentence_sentiments(subset: pd.DataFrame):
    subset[['content_sentiment_sentence_sub', 'content_sentiment_sentence_pos', 'content_sentiment_sentence_neg']] = subset[
        'content_sentences_no_stopwords'].apply(get_sentiment_from_tokens)
    subset[['title_sentiment_sentence_sub', 'title_sentiment_sentence_pos', 'title_sentiment_sentence_neg']] = subset[
        'title_sentences_no_stopwords'].apply(get_sentiment_from_tokens)
    return subset


def text_sentiments(subset: pd.DataFrame):
    subset[['content_sentiment_text_sub', 'content_sentiment_text_pos', 'content_sentiment_text_neg']] = subset[
        'content_no_stopwords'].apply(get_sentiment_from_text)
    subset[['title_sentiment_text_sub', 'title_sentiment_text_pos', 'title_sentiment_text_neg']] = subset[
        'title_no_stopwords'].apply(get_sentiment_from_text)
    return subset
