from snorkel.labeling import labeling_function
from snorkel_weak_labeling.manual.thresholds import *

"""

@labeling_function()
def lf_(x):
    return REAL if x. > else ABSTAIN

"""


""" Content text sentiment """


@labeling_function()
def lf_content_text_is_negative(x):
    return FAKE if x.content_sentiment_text_neg > CONTENT_SENTIMENT_TEXT_NEG_UPPER else ABSTAIN


# @labeling_function()
# def lf_content_text_not_negative(x):
#     return REAL if x.content_sentiment_text_neg <= CONTENT_SENTIMENT_TEXT_NEG_LOWER else ABSTAIN


# @labeling_function()
# def lf_content_text_is_positive(x):
#     return FAKE if x.content_sentiment_text_pos > CONTENT_SENTIMENT_TEXT_POS_UPPER else ABSTAIN


@labeling_function()
def lf_content_text_not_positive(x):
    return FAKE if x.content_sentiment_text_pos <= CONTENT_SENTIMENT_TEXT_POS_LOWER else ABSTAIN


@labeling_function()
def lf_content_text_is_subjective(x):
    return FAKE if x.content_sentiment_text_sub > CONTENT_SENTIMENT_TEXT_SUB_UPPER else ABSTAIN


@labeling_function()
def lf_content_text_not_subjective(x):
    return REAL if x.content_sentiment_text_sub <= CONTENT_SENTIMENT_TEXT_SUB_LOWER else ABSTAIN


""" Content sentence sentiment """


@labeling_function()
def lf_content_sentence_is_negative(x):
    return REAL if x.content_sentiment_sentence_neg > CONTENT_SENTIMENT_SENTENCE_NEG_UPPER else ABSTAIN


@labeling_function()
def lf_content_sentence_not_negative(x):
    return FAKE if x.content_sentiment_sentence_neg <= CONTENT_SENTIMENT_SENTENCE_NEG_LOWER else ABSTAIN


@labeling_function()
def lf_content_sentence_is_positive(x):
    return FAKE if x.content_sentiment_sentence_pos > CONTENT_SENTIMENT_SENTENCE_POS_UPPER else ABSTAIN


@labeling_function()
def lf_content_sentence_not_positive(x):
    return REAL if x.content_sentiment_sentence_pos <= CONTENT_SENTIMENT_SENTENCE_POS_LOWER else ABSTAIN


@labeling_function()
def lf_content_sentence_is_subjective(x):
    return FAKE if x.content_sentiment_sentence_sub > CONTENT_SENTIMENT_SENTENCE_SUB_UPPER else ABSTAIN


@labeling_function()
def lf_content_sentence_not_subjective(x):
    return REAL if x.content_sentiment_sentence_sub <= CONTENT_SENTIMENT_SENTENCE_SUB_LOWER else ABSTAIN


""" Content word sentiment """


# @labeling_function()
# def lf_content_word_is_negative(x):
#     return REAL if x.content_sentiment_word_neg > CONTENT_SENTIMENT_WORD_NEG_UPPER else ABSTAIN


# @labeling_function()
# def lf_content_word_not_negative(x):
#     return REAL if x.content_sentiment_word_neg <= CONTENT_SENTIMENT_WORD_NEG_LOWER else ABSTAIN


# @labeling_function()
# def lf_content_word_is_positive(x):
#     return FAKE if x.content_sentiment_word_pos > CONTENT_SENTIMENT_WORD_POS_UPPER else ABSTAIN
#
#
# @labeling_function()
# def lf_content_word_not_positive(x):
#     return REAL if x.content_sentiment_word_pos <= CONTENT_SENTIMENT_WORD_POS_LOWER else ABSTAIN


@labeling_function()
def lf_content_word_is_subjective(x):
    return FAKE if x.content_sentiment_word_sub > CONTENT_SENTIMENT_WORD_SUB_UPPER else ABSTAIN


@labeling_function()
def lf_content_word_not_subjective(x):
    return REAL if x.content_sentiment_word_sub <= CONTENT_SENTIMENT_WORD_SUB_LOWER else ABSTAIN


""" Title text sentiment """


@labeling_function()
def lf_title_text_is_negative(x):
    return FAKE if x.title_sentiment_text_neg > TITLE_SENTIMENT_TEXT_NEG_UPPER else ABSTAIN


@labeling_function()
def lf_title_text_not_negative(x):
    return FAKE if x.title_sentiment_text_neg <= TITLE_SENTIMENT_TEXT_NEG_LOWER else ABSTAIN


# @labeling_function()
# def lf_title_text_is_positive(x):
#     return FAKE if x.title_sentiment_text_pos > TITLE_SENTIMENT_TEXT_POS_UPPER else ABSTAIN
#
#
# @labeling_function()
# def lf_title_text_not_positive(x):
#     return REAL if x.title_sentiment_text_pos <= TITLE_SENTIMENT_TEXT_POS_LOWER else ABSTAIN


@labeling_function()
def lf_title_text_is_subjective(x):
    return FAKE if x.title_sentiment_text_sub > TITLE_SENTIMENT_TEXT_SUB_UPPER else ABSTAIN


# @labeling_function()
# def lf_title_text_not_subjective(x):
#     return REAL if x.title_sentiment_text_sub <= TITLE_SENTIMENT_TEXT_SUB_LOWER else ABSTAIN


""" Title sentence sentiment """


# @labeling_function()
# def lf_title_sentence_is_negative(x):
#     return FAKE if x.title_sentiment_sentence_neg > TITLE_SENTIMENT_SENTENCE_NEG_UPPER else ABSTAIN
#
#
# @labeling_function()
# def lf_title_sentence_not_negative(x):
#     return REAL if x.title_sentiment_sentence_neg <= TITLE_SENTIMENT_SENTENCE_NEG_LOWER else ABSTAIN


# @labeling_function()
# def lf_title_sentence_is_positive(x):
#     return FAKE if x.title_sentiment_sentence_pos > TITLE_SENTIMENT_SENTENCE_POS_UPPER else ABSTAIN
#
#
# @labeling_function()
# def lf_title_sentence_not_positive(x):
#     return REAL if x.title_sentiment_sentence_pos <= TITLE_SENTIMENT_SENTENCE_POS_LOWER else ABSTAIN


@labeling_function()
def lf_title_sentence_is_subjective(x):
    return FAKE if x.title_sentiment_sentence_sub > TITLE_SENTIMENT_SENTENCE_SUB_UPPER else ABSTAIN


# @labeling_function()
# def lf_title_sentence_not_subjective(x):
#     return REAL if x.title_sentiment_sentence_sub <= TITLE_SENTIMENT_SENTENCE_SUB_LOWER else ABSTAIN


""" Title word sentiment """


@labeling_function()
def lf_title_word_is_negative(x):
    return FAKE if x.title_sentiment_word_neg > TITLE_SENTIMENT_WORD_NEG_UPPER else ABSTAIN


@labeling_function()
def lf_title_word_not_negative(x):
    return FAKE if x.title_sentiment_word_neg <= TITLE_SENTIMENT_WORD_NEG_LOWER else ABSTAIN


@labeling_function()
def lf_title_word_is_positive(x):
    return FAKE if x.title_sentiment_word_pos > TITLE_SENTIMENT_WORD_POS_UPPER else ABSTAIN


@labeling_function()
def lf_title_word_not_positive(x):
    return FAKE if x.title_sentiment_word_pos <= TITLE_SENTIMENT_WORD_POS_LOWER else ABSTAIN


@labeling_function()
def lf_title_word_is_subjective(x):
    return FAKE if x.title_sentiment_word_sub > TITLE_SENTIMENT_WORD_SUB_UPPER else ABSTAIN


@labeling_function()
def lf_title_word_not_subjective(x):
    return FAKE if x.title_sentiment_word_sub <= TITLE_SENTIMENT_WORD_SUB_LOWER else ABSTAIN



