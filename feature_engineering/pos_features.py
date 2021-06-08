# -*- coding: utf-8 -*-
import pandas as pd
import nltk
from collections import Counter


def pos_tagging(subset: pd.DataFrame):
    subset['content_pos_tags'] = subset['content_tokenized_words'].apply(nltk.pos_tag)
    subset['content_pos_tags_no_stopwords'] = subset['content_words_no_stopwords'].apply(nltk.pos_tag)
    subset['title_pos_tags'] = subset['title_tokenized_words'].apply(nltk.pos_tag)
    subset['title_pos_tags_lowercase'] = subset['title_tokenized_words_lowercase'].apply(nltk.pos_tag)
    subset['title_pos_tags_no_stopwords'] = subset['title_words_no_stopwords'].apply(nltk.pos_tag)
    subset['title_pos_tags_no_stopwords_lowercase'] = subset['title_words_no_stopwords_lowercase'].apply(nltk.pos_tag)
    return subset


"""
Article level features
"""


def count_personal_pronouns(subset: pd.DataFrame):

    def count_prp(pos_tags):
        counts = Counter(tag for word, tag in pos_tags)

        if len(pos_tags):
            return pd.Series([counts['PRP'], counts['PRP']/len(pos_tags)])
        return pd.Series([counts['PRP'], 0])

    subset[["content_personal_pronouns_count", "content_personal_pronouns_ratio"]] = subset['content_pos_tags'].apply(
        count_prp)
    return subset


def count_verbs_and_adjectives(subset: pd.DataFrame):

    def count(pos_tags):
        counts = Counter(tag for word, tag in pos_tags)

        verb_counts = counts['VB'] + counts['VBD'] + counts['VBG'] + counts['VBN'] + counts['VBP'] + counts['VBZ']
        verb_past_tense_counts = counts['VBD'] + counts['VBN']
        adjective_count = counts['JJ'] + counts['JJR'] + counts['JJS']
        adverb_count = counts['RB'] + counts['RBR'] + counts['RBS']

        total_words = len(pos_tags)

        past_tense_ratio = 0
        if verb_counts != 0:
            past_tense_ratio = verb_past_tense_counts/verb_counts

        if total_words:
            return pd.Series([verb_counts/total_words,
                              verb_past_tense_counts/total_words,
                              past_tense_ratio,
                              adjective_count/total_words,
                              adverb_count/total_words
                              ])
        return pd.Series([0, 0, past_tense_ratio, 0, 0])

    subset[['content_verb_ratio',
            'content_past_tense_verb_ratio',
            'content_past_tense_verb_ratio_of_all_verbs',
            'content_adjective_ratio',
            'content_adverb_ratio']] = subset['content_pos_tags'].apply(count)
    subset[['title_verb_ratio',
            'title_past_tense_verb_ratio',
            'title_past_tense_verb_ratio_of_all_verbs',
            'title_adjective_ratio',
            'title_adverb_ratio']] = subset['title_pos_tags_lowercase'].apply(count)
    return subset


"""
Title level features
"""


def title_count_pos(subset: pd.DataFrame):
    # Creates list with pos tag counts e.g. [(NNP, 8), (CD, 1), (RB, 1), (VBN, 1)]

    def count_pos(pos_tags):
        counts = Counter(tag for word, tag in pos_tags)

        return list(counts.items())

    subset["title_count_pos"] = subset['title_pos_tags'].apply(count_pos)

    return subset


def title_count_nouns(subset: pd.DataFrame):
    # Counts all types of nouns

    noun_types = ["NN", "NNP", "NNS", "NNPS"]

    def count_nouns(pos_list):

        noun_count = 0
        total_count = 0
        for pos_tuple in pos_list:
            if pos_tuple[0] in noun_types:
                noun_count += pos_tuple[1]
            total_count += pos_tuple[1]

        if total_count:
            return pd.Series([noun_count, noun_count/total_count])
        return pd.Series([noun_count, 0])

    subset[["title_nouns_count", 'title_nouns_ratio']] = subset["title_count_pos"].apply(count_nouns)

    return subset


def title_count_proper_nouns(subset: pd.DataFrame):
    # Counts only proper nouns (e.g. names)

    proper_noun_types = ["NNP", "NNPS"]

    def count_proper_nouns(pos_list):

        proper_noun_count = 0
        total_count = 0
        for pos_tuple in pos_list:
            if pos_tuple[0] in proper_noun_types:
                proper_noun_count += pos_tuple[1]
            total_count += pos_tuple[1]

        if total_count:
            return pd.Series([proper_noun_count, proper_noun_count/total_count])
        return pd.Series([proper_noun_count, 0])

    subset[["title_proper_nouns_count", "title_proper_nouns_ratio"]] = subset["title_count_pos"].apply(
        count_proper_nouns)

    return subset

