from snorkel.labeling import labeling_function
from snorkel_weak_labeling.manual.thresholds import *


""" TTR score """


@labeling_function()
def lf_content_ttr_score_high(x):
    return FAKE if (CONTENT_TTR_SCORE_FAKE_MID_UPPER < x.content_ttr_score < CONTENT_TTR_SCORE_FAKE_UPPER) else ABSTAIN


# @labeling_function()
# def lf_content_ttr_score_mid(x):
#     return REAL if (CONTENT_TTR_SCORE_REAL_LOWER < x.content_ttr_score < CONTENT_TTR_SCORE_REAL_UPPER) else ABSTAIN


@labeling_function()
def lf_content_ttr_score_low(x):
    return FAKE if (CONTENT_TTR_SCORE_FAKE_LOWER < x.content_ttr_score < CONTENT_TTR_SCORE_FAKE_MID_LOWER) else ABSTAIN


# @labeling_function()
# def lf_title_ttr_score_high(x):
#     return FAKE if x.title_ttr_score > TITLE_TTR_SCORE_UPPER else ABSTAIN
#
#
# @labeling_function()
# def lf_title_ttr_score_low(x):
#     return REAL if x.title_ttr_score <= TITLE_TTR_SCORE_LOWER else ABSTAIN


""" Average words per sentence """


@labeling_function()
def lf_content_avg_words_per_sentence_high(x):
    return FAKE if x.content_words_per_sentence > CONTENT_WORDS_PER_SENTENCE_UPPER else ABSTAIN


# @labeling_function()
# def lf_content_avg_words_per_sentence_low(x):
#     return REAL if x.content_words_per_sentence <= CONTENT_WORDS_PER_SENTENCE_LOWER else ABSTAIN


""" Average word length """


@labeling_function()
def lf_content_avg_word_length_high(x):
    return REAL if x.content_avg_word_length > CONTENT_AVG_WORD_LENGTH_REAL_UPPER else ABSTAIN


@labeling_function()
def lf_content_avg_word_length_medium_high(x):
    return FAKE if (CONTENT_AVG_WORD_LENGTH_FAKE_UPPER < x.content_avg_word_length <
                    CONTENT_AVG_WORD_LENGTH_FAKE_MID_UPPER) else ABSTAIN


@labeling_function()
def lf_content_avg_word_length_medium_low(x):
    return REAL if (CONTENT_AVG_WORD_LENGTH_REAL_LOWER < x.content_avg_word_length <
                    CONTENT_AVG_WORD_LENGTH_REAL_MID) else ABSTAIN


@labeling_function()
def lf_content_avg_word_length_low(x):
    return FAKE if (CONTENT_AVG_WORD_LENGTH_FAKE_LOWER < x.content_avg_word_length <
                    CONTENT_AVG_WORD_LENGTH_FAKE_MID) else ABSTAIN


# @labeling_function()
# def lf_title_avg_word_length_high(x):
#     return FAKE if (TITLE_AVG_WORD_LENGTH_MID < x.title_avg_word_length < TITLE_AVG_WORD_LENGTH_UPPER) else ABSTAIN
#
#
# @labeling_function()
# def lf_title_avg_word_length_low(x):
#     return REAL if x.title_avg_word_length <= TITLE_AVG_WORD_LENGTH_LOWER else ABSTAIN


""" Average word length no stop words """


@labeling_function()
def lf_content_avg_word_length_no_stop_words_high(x):
    return FAKE if x.content_avg_word_length_no_stop_words > CONTENT_AVG_WORD_LENGTH_NO_STOP_WORDS_FAKE_UPPER else ABSTAIN


@labeling_function()
def lf_content_avg_word_length_no_stop_words_mid(x):
    return REAL if (CONTENT_AVG_WORD_LENGTH_NO_STOP_WORDS_REAL_LOWER < x.content_avg_word_length_no_stop_words <
                    CONTENT_AVG_WORD_LENGTH_NO_STOP_WORDS_REAL_UPPER) else ABSTAIN


@labeling_function()
def lf_content_avg_word_length_no_stop_words_low(x):
    return FAKE if (CONTENT_AVG_WORD_LENGTH_NO_STOP_WORDS_FAKE_LOWER < x.content_avg_word_length_no_stop_words <
                    CONTENT_AVG_WORD_LENGTH_NO_STOP_WORDS_FAKE_MID) else ABSTAIN


# @labeling_function()
# def lf_title_avg_word_length_no_stop_words_high(x):
#     return FAKE if x.title_avg_word_length_no_stop_words > TITLE_AVG_WORD_LENGTH_NO_STOP_WORDS_UPPER else ABSTAIN
#

@labeling_function()
def lf_title_avg_word_length_no_stop_words_low(x):
    return FAKE if x.title_avg_word_length_no_stop_words <= TITLE_AVG_WORD_LENGTH_NO_STOP_WORDS_LOWER else ABSTAIN
