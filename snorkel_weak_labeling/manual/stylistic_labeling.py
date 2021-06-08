from snorkel.labeling import labeling_function
from snorkel_weak_labeling.manual.thresholds import *

"""

@labeling_function()
def lf_(x):
    return REAL if x. > else ABSTAIN
    
"""


""" Exclamation marks """


@labeling_function()
def lf_content_exclamation_count_fake(x):
    return FAKE if (x.content_exclamation_count > CONTENT_EXCLAMATION_LOWER) and (
        x.content_exclamation_count <= CONTENT_EXCLAMATION_UPPER
    ) else ABSTAIN


@labeling_function()
def lf_content_exclamation_count_real(x):
    return REAL if (x.content_exclamation_count > CONTENT_EXCLAMATION_UPPER) and (
        x.content_exclamation_count <= CONTENT_EXCLAMATION_LOWER
    ) else ABSTAIN


@labeling_function()
def lf_content_exclamation_ratio_high(x):
    return REAL if x.content_exclamation_ratio > CONTENT_EXCLAMATION_RATIO_UPPER else ABSTAIN


@labeling_function()
def lf_content_exclamation_ratio_low(x):
    return FAKE if (x.content_exclamation_ratio <= CONTENT_EXCLAMATION_RATIO_UPPER) and (
        x.content_exclamation_ratio > CONTENT_EXCLAMATION_RATIO_LOWER
    ) else ABSTAIN


@labeling_function()
def lf_title_has_exclamation_marks(x):
    return REAL if x.title_exclamation_count > TITLE_EXCLAMATION_UPPER else ABSTAIN


@labeling_function()
def lf_title_no_exclamation_marks(x):
    return FAKE if (x.title_exclamation_count <= TITLE_EXCLAMATION_UPPER) and (
        x.content_exclamation_count >= TITLE_EXCLAMATION_LOWER
    ) else ABSTAIN


@labeling_function()
def lf_title_exclamation_ratio_high(x):
    return FAKE if (x.title_exclamation_ratio > TITLE_EXCLAMATION_RATIO_LOWER) and (
        x.title_exclamation_ratio <= TITLE_EXCLAMATION_RATIO_UPPER
    ) else ABSTAIN


@labeling_function()
def lf_title_exclamation_ratio_low(x):
    return FAKE if x.title_exclamation_ratio > TITLE_EXCLAMATION_RATIO_UPPER else ABSTAIN


""" Word counts """


@labeling_function()
def lf_content_word_count_high(x):
    return REAL if x.content_word_count > CONTENT_WORD_COUNT_UPPER else ABSTAIN


@labeling_function()
def lf_content_word_count_low(x):
    return FAKE if x.content_word_count <= CONTENT_WORD_COUNT_LOWER else ABSTAIN


@labeling_function()
def lf_title_word_count_high(x):
    return FAKE if x.title_word_count > TITLE_WORD_COUNT_UPPER else ABSTAIN


@labeling_function()
def lf_title_word_count_low(x):
    return FAKE if x.title_word_count <= TITLE_WORD_COUNT_LOWER else ABSTAIN


""" Sentence count """


@labeling_function()
def lf_content_sentence_count_high(x):
    return REAL if x.content_sentence_count > CONTENT_SENTENCE_COUNT_UPPER else ABSTAIN


@labeling_function()
def lf_content_sentence_count_low(x):
    return FAKE if x.content_sentence_count <= CONTENT_SENTENCE_COUNT_LOWER else ABSTAIN


@labeling_function()
def lf_title_sentence_count_high(x):
    return FAKE if x.title_sentence_count > TITLE_SENTENCE_COUNT_UPPER else ABSTAIN


# @labeling_function()
# def lf_title_sentence_count_low(x):
#     return FAKE if x.title_sentence_count <= TITLE_SENTENCE_COUNT_LOWER else ABSTAIN


""" URLs """


@labeling_function()
def lf_content_url_count_high(x):
    return REAL if x.content_url_count > CONTENT_URL_COUNT_UPPER else ABSTAIN


# @labeling_function()
# def lf_content_url_count_low(x):
#     return REAL if x.content_url_count <= CONTENT_URL_COUNT_LOWER else ABSTAIN


""" Quote marks """


@labeling_function()
def lf_content_quotation_count_high(x):
    return FAKE if x.content_quote_marks_count > CONTENT_QUOTE_COUNT_UPPER else ABSTAIN


# @labeling_function()
# def lf_content_quotation_count_low(x):
#     return REAL if x.content_quote_marks_count <= CONTENT_QUOTE_COUNT_LOWER else ABSTAIN


@labeling_function()
def lf_content_quotation_ratio_high(x):
    return FAKE if x.content_quote_marks_ratio > CONTENT_QUOTE_RATIO_UPPER else ABSTAIN


# @labeling_function()
# def lf_content_quotation_ratio_low(x):
#     return REAL if x.content_quote_marks_ratio <= CONTENT_QUOTE_RATIO_LOWER else ABSTAIN


""" Stop word count """


@labeling_function()
def lf_content_stop_word_count_high(x):
    return REAL if x.content_stop_word_count > CONTENT_STOP_WORD_COUNT_UPPER else ABSTAIN


@labeling_function()
def lf_content_stop_word_count_low(x):
    return FAKE if x.content_stop_word_count <= CONTENT_STOP_WORD_COUNT_LOWER else ABSTAIN


@labeling_function()
def lf_title_stop_word_count_high(x):
    return REAL if x.title_stop_word_count > TITLE_STOP_WORD_COUNT_UPPER else ABSTAIN


@labeling_function()
def lf_title_stop_word_count_low(x):
    return FAKE if x.title_stop_word_count <= TITLE_STOP_WORD_COUNT_LOWER else ABSTAIN


""" Stop word ratio """


@labeling_function()
def lf_content_stop_word_ratio_high(x):
    return FAKE if x.content_stop_word_ratio > CONTENT_STOP_WORD_RATIO_UPPER else ABSTAIN


# @labeling_function()
# def lf_content_stop_word_ratio_low(x):
#     return REAL if x.content_stop_word_ratio <= CONTENT_STOP_WORD_RATIO_LOWER else ABSTAIN


@labeling_function()
def lf_title_stop_word_ratio_high(x):
    return REAL if x.title_stop_word_ratio > TITLE_STOP_WORD_RATIO_UPPER else ABSTAIN


@labeling_function()
def lf_title_stop_word_ratio_low(x):
    return REAL if x.title_stop_word_ratio <= TITLE_STOP_WORD_RATIO_LOWER else ABSTAIN


""" Captial word count """


@labeling_function()
def lf_content_capital_word_count_high(x):
    return FAKE if x.content_capital_word_count > CONTENT_CAPITAL_WORD_COUNT_UPPER else ABSTAIN


# @labeling_function()
# def lf_content_capital_word_count_low(x):
#     return REAL if x.content_capital_word_count <= CONTENT_CAPITAL_WORD_COUNT_LOWER else ABSTAIN


@labeling_function()
def lf_title_capital_word_count_high(x):
    return FAKE if x.title_capital_word_count > TITLE_CAPITAL_WORD_COUNT_UPPER else ABSTAIN


# @labeling_function()
# def lf_title_capital_word_count_low(x):
#     return REAL if x.title_capital_word_count <= TITLE_CAPITAL_WORD_COUNT_LOWER else ABSTAIN


""" Capital word ratio """


@labeling_function()
def lf_content_capital_word_ratio_high(x):
    return FAKE if x.content_capital_word_ratio > CONTENT_CAPITAL_WORD_RATIO_UPPER else ABSTAIN


@labeling_function()
def lf_content_capital_word_ratio_low(x):
    return REAL if x.content_capital_word_ratio <= CONTENT_CAPITAL_WORD_RATIO_LOWER else ABSTAIN


@labeling_function()
def lf_title_capital_word_ratio_high(x):
    return FAKE if x.title_capital_word_ratio > TITLE_CAPITAL_WORD_RATIO_UPPER else ABSTAIN


# @labeling_function()
# def lf_title_capital_word_ratio_low(x):
#     return REAL if x.title_capital_word_ratio <= TITLE_CAPITAL_WORD_RATIO_LOWER else ABSTAIN