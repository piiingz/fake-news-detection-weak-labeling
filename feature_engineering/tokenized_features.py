# -*- coding: utf-8 -*-
import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

PUNCTUATION = string.punctuation + '“”‘’-—–'

cleaning_regex = r"\w+s['´’]|\w+[-'´’.]?\w+|\w"

lemmatizer = WordNetLemmatizer()


def word_tokenize(subset: pd.DataFrame):

    def clean(article):
        article_chars = re.findall(cleaning_regex, str(article))
        cleaned_article = (" ".join(article_chars))
        tokens = cleaned_article.split()
        return tokens

    subset['content_tokenized_words'] = subset['content'].apply(clean)
    subset['content_tokenized_words_with_punctuation'] = subset['content'].apply(lambda x: nltk.word_tokenize(str(x)))
    subset['title_tokenized_words'] = subset['title'].apply(clean)
    subset['title_tokenized_words_with_punctuation'] = subset['title'].apply(lambda x: nltk.word_tokenize(str(x)))
    return subset


def sentence_tokenize(subset: pd.DataFrame):

    def tokenize(article):
        paragraphs = article.split('\n')
        tokens = []
        for p in paragraphs:
            tokens.extend(nltk.sent_tokenize(p))
        return tokens

    def clean(article):
        tokens = tokenize(str(article))
        sentences = []
        for sentence in tokens:
            stripped_sentence = sentence.rstrip('\n')
            sentence_chars = re.findall(cleaning_regex, stripped_sentence)
            cleaned_sentence = (" ".join(sentence_chars))
            sentences.append(cleaned_sentence)
        return sentences

    subset['content_tokenized_sentences'] = subset['content'].apply(clean)
    subset['title_tokenized_sentences'] = subset['title'].apply(clean)
    return subset


def lowercase(subset: pd.DataFrame):
    for col in ['content_tokenized_words']:
        feature_name = col + '_lowercase'
        subset[feature_name] = subset[col].apply(lambda x: [word.lower() for word in x])
    for col in ['content_tokenized_sentences']:
        feature_name = col + '_lowercase'
        subset[feature_name] = subset[col].apply(lambda x: [word.lower() for word in x])
    for col in ['title_tokenized_words', 'title_words_no_stopwords']:
        feature_name = col + '_lowercase'
        subset[feature_name] = subset[col].apply(lambda x: [word.lower() for word in x])
    for col in ['title_tokenized_sentences']:
        feature_name = col + '_lowercase'
        subset[feature_name] = subset[col].apply(lambda x: [word.lower() for word in x])
    return subset


def remove_column(subset: pd.DataFrame):
    subset = subset.drop("tokenized_word_with_punctuation", 1)
    return subset


def remove_punctuation(subset: pd.DataFrame):
    def clean(article):
        article.rstrip("\n")
        cleaned = "".join([char for char in str(article) if char not in PUNCTUATION])
        return cleaned

    subset['content_no_punctuation'] = subset['content'].apply(clean)
    subset['title_no_punctuation'] = subset['title'].apply(clean)
    return subset


def remove_stop_words(subset: pd.DataFrame):

    def remove_all_stop_words(tokens):
        stop_words = stopwords.words('english')
        return [word for word in tokens if (word.lower() not in stop_words)]

    def remove_all_stop_words_sentences(sentences):
        return [" ".join(remove_all_stop_words(sentence.split(" "))) for sentence in sentences]

    subset['content_words_no_stopwords'] = subset['content_tokenized_words'].apply(remove_all_stop_words)
    subset['title_words_no_stopwords'] = subset['title_tokenized_words'].apply(remove_all_stop_words)

    subset['content_sentences_no_stopwords'] = subset['content_tokenized_sentences'].apply(remove_all_stop_words_sentences)
    subset['title_sentences_no_stopwords'] = subset['title_tokenized_sentences'].apply(remove_all_stop_words_sentences)

    subset['content_no_stopwords'] = subset['content_sentences_no_stopwords'].apply(lambda x: " ".join(x))
    subset['title_no_stopwords'] = subset['title_sentences_no_stopwords'].apply(lambda x: " ".join(x))

    return subset


# For converting between treebank pos tags (1st character) to wordnet pos, uses NOUN as default
def get_wordnet_pos(pos_first_letter):
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(pos_first_letter, wordnet.NOUN)


def lemmatize_tokens(subset: pd.DataFrame):

    def lemmatize_article(pos_tagged_words):

        lemmatized_words = []

        for pos_tuple in pos_tagged_words:
            word = pos_tuple[0]
            tag_first_letter = pos_tuple[1][0]
            wordnet_tag = get_wordnet_pos(tag_first_letter)

            lemma = lemmatizer.lemmatize(word, wordnet_tag)
            lemmatized_words.append(lemma)

        return lemmatized_words

    subset['content_lemmatized_lowercase_no_stopwords'] = subset['content_pos_tags_no_stopwords'].apply(lemmatize_article)
    subset['title_lemmatized_lowercase_no_stopwords'] = subset['title_pos_tags_no_stopwords_lowercase'].apply(lemmatize_article)

    return subset
