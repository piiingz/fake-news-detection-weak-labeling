from snorkel.labeling import labeling_function
from snorkel_weak_labeling.manual.thresholds import *


""" Adjectives """


@labeling_function()
def lf_content_adjective_ratio_high(x):
    return FAKE if x.content_adjective_ratio > CONTENT_ADJECTIVE_RATIO_UPPER else ABSTAIN


@labeling_function()
def lf_content_adjective_ratio_low(x):
    return FAKE if x.content_adjective_ratio <= CONTENT_ADJECTIVE_RATIO_LOWER else ABSTAIN


@labeling_function()
def lf_title_adjective_ratio_high(x):
    return REAL if x.title_adjective_ratio > TITLE_ADJECTIVE_RATIO_UPPER else ABSTAIN


# @labeling_function()
# def lf_title_adjective_ratio_low(x):
#     return REAL if x.title_adjective_ratio <= TITLE_ADJECTIVE_RATIO_LOWER else ABSTAIN


""" Personal pronouns"""


@labeling_function()
def lf_content_personal_pronouns_count_high(x):
    return REAL if x.content_personal_pronouns_count > CONTENT_PERSONAL_PRONOUNS_COUNT_UPPER else ABSTAIN


@labeling_function()
def lf_content_personal_pronouns_count_low(x):
    return REAL if x.content_personal_pronouns_count <= CONTENT_PERSONAL_PRONOUNS_COUNT_LOWER else ABSTAIN


@labeling_function()
def lf_content_personal_pronouns_ratio_high(x):
    return REAL if x.content_personal_pronouns_ratio > CONTENT_PERSONAL_PRONOUNS_RATIO_UPPER else ABSTAIN


@labeling_function()
def lf_content_personal_pronouns_ratio_low(x):
    return REAL if x.content_personal_pronouns_ratio <= CONTENT_PERSONAL_PRONOUNS_RATIO_LOWER else ABSTAIN


""" Verbs """


@labeling_function()
def lf_content_verb_ratio_high(x):
    return FAKE if x.content_verb_ratio > CONTENT_VERB_RATIO_UPPER else ABSTAIN


@labeling_function()
def lf_content_verb_ratio_low(x):
    return FAKE if x.content_verb_ratio <= CONTENT_VERB_RATIO_LOWER else ABSTAIN


@labeling_function()
def lf_title_verb_ratio_high(x):
    return FAKE if x.title_verb_ratio > TITLE_VERB_RATIO_UPPER else ABSTAIN


@labeling_function()
def lf_title_verb_ratio_low(x):
    return REAL if x.title_verb_ratio <= TITLE_VERB_RATIO_LOWER else ABSTAIN


""" Past tense verbs / all words """


@labeling_function()
def lf_content_past_tense_verb_ratio_high(x):
    return REAL if x.content_past_tense_verb_ratio > CONTENT_PAST_TENSE_VERB_RATIO_UPPER else ABSTAIN


@labeling_function()
def lf_content_past_tense_verb_ratio_mid(x):
    return FAKE if (CONTENT_PAST_TENSE_VERB_RATIO_MID < x.content_past_tense_verb_ratio
                    < CONTENT_PAST_TENSE_VERB_RATIO_UPPER) else ABSTAIN


@labeling_function()
def lf_content_past_tense_verb_ratio_low(x):
    return REAL if (CONTENT_PAST_TENSE_VERB_RATIO_LOWER < x.content_past_tense_verb_ratio
                    < CONTENT_PAST_TENSE_VERB_RATIO_MID) else ABSTAIN


@labeling_function()
def lf_title_past_tense_verb_ratio_high(x):
    return FAKE if x.title_past_tense_verb_ratio > TITLE_PAST_TENSE_VERB_RATIO_UPPER else ABSTAIN


# @labeling_function()
# def lf_title_past_tense_verb_ratio_low(x):
#     return REAL if x.title_past_tense_verb_ratio <= TITLE_PAST_TENSE_VERB_RATIO_LOWER else ABSTAIN


""" Past tense verbs / no of verbs"""


@labeling_function()
def lf_content_past_tense_verb_ratio_of_all_verbs_high(x):
    return REAL if (x.content_past_tense_verb_ratio_of_all_verbs >
                    CONTENT_PAST_TENSE_VERB_OF_ALL_VERBS_RATIO_UPPER) else ABSTAIN


@labeling_function()
def lf_content_past_tense_verb_ratio_of_all_verbs_low(x):
    return FAKE if (CONTENT_PAST_TENSE_VERB_OF_ALL_VERBS_RATIO_LOWER < x.content_past_tense_verb_ratio_of_all_verbs
                   < CONTENT_PAST_TENSE_VERB_OF_ALL_VERBS_RATIO_MID) else ABSTAIN


# @labeling_function()
# def lf_title_past_tense_verb_ratio_of_all_verbs_high(x):
#     return FAKE if (x.title_past_tense_verb_ratio_of_all_verbs >
#                     TITLE_PAST_TENSE_VERB_OF_ALL_VERBS_RATIO_UPPER) else ABSTAIN
#
#
# @labeling_function()
# def lf_title_past_tense_verb_ratio_of_all_verbs_low(x):
#     return REAL if (x.title_past_tense_verb_ratio_of_all_verbs <=
#                     TITLE_PAST_TENSE_VERB_OF_ALL_VERBS_RATIO_LOWER) else ABSTAIN


""" Nouns """


@labeling_function()
def lf_title_nouns_count_high(x):
    return FAKE if x.title_nouns_count > TITLE_NOUNS_COUNT_UPPER else ABSTAIN


@labeling_function()
def lf_title_nouns_count_low(x):
    return REAL if x.title_nouns_count <= TITLE_NOUNS_COUNT_LOWER else ABSTAIN


@labeling_function()
def lf_title_nouns_ratio_high(x):
    return FAKE if x.title_nouns_ratio > TITLE_NOUNS_RATIO_LOWER else ABSTAIN


@labeling_function()
def lf_title_nouns_ratio_low(x):
    return REAL if x.title_nouns_ratio <= TITLE_NOUNS_RATIO_LOWER else ABSTAIN


""" Proper nouns """


@labeling_function()
def lf_title_proper_nouns_count_high(x):
    return FAKE if x.title_proper_nouns_count > TITLE_PROPER_NOUNS_COUNT_UPPER else ABSTAIN


@labeling_function()
def lf_title_proper_nouns_count_low(x):
    return REAL if x.title_proper_nouns_count <= TITLE_PROPER_NOUNS_COUNT_LOWER else ABSTAIN


@labeling_function()
def lf_title_proper_nouns_ratio_high(x):
    return FAKE if x.title_proper_nouns_ratio > TITLE_PROPER_NOUNS_RATIO_UPPER else ABSTAIN


@labeling_function()
def lf_title_proper_nouns_ratio_low(x):
    return REAL if x.title_proper_nouns_ratio <= TITLE_PROPER_NOUNS_RATIO_LOWER else ABSTAIN
