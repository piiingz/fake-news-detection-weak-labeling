""" Global parameters """
IDUN = True

RANDOM_SEED = 2021
TEST_SIZE = 0.2

FAKE = 1
REAL = 0
ABSTAIN = -1

SPLIT_SIZE = 100000

""" Threshold generation """
UPPER_QUANTILES = ['70%', '75%', '80%', '85%', '90%', '95%']
LOWER_QUANTILES = ['30%', '25%', '20%', '15%', '10%', '5%']

""" Apply weak labels """
NR_OF_LFS = 25
ACC_OF_LFS = 0.65
AUTOMATED_LFS = True

""" Snuba """
NUM_TEST = 0
VAL_RATIO = 0.25
GOAL_MAX_CARDINALITY = 3
MODEL_TYPES = {
    1: 'dt',
    2: 'lr',
    3: 'nn'
}
KEEP = 3
IS_LOCAL = False

""" Lists """
LFS_ABOVE_60 = ['lf_content_exclamation_ratio_high',
                'lf_content_stop_word_ratio_high'
                'lf_title_word_is_negative',
                'lf_title_word_is_positive',
                'lf_title_word_not_negative',
                'lf_title_word_not_positive',
                'lf_title_sentence_count_high',
                'lf_title_proper_nouns_count_high',
                'lf_content_verb_ratio_low',
                'lf_title_word_count_low',
                'lf_title_proper_nouns_ratio_high',
                'lf_content_exclamation_ratio_low',
                'lf_title_nouns_count_high',
                'lf_content_exclamation_count_fake',
                'lf_title_no_exclamation_marks',
                'lf_title_proper_nouns_ratio_low',
                'lf_title_capital_word_count_high',
                'lf_title_capital_word_ratio_high',
                'lf_title_nouns_ratio_high',
                'lf_title_proper_nouns_count_low',
                'lf_content_avg_word_length_high',
                'lf_content_avg_word_length_low',
                'lf_content_avg_word_length_medium_low',
                'lf_content_avg_word_length_no_stop_words_high',
                'lf_content_avg_word_length_no_stop_words_low',
                'lf_content_avg_words_per_sentence_high'
                ]

LFS_ABOVE_65 = ['lf_content_exclamation_ratio_high',
                'lf_content_stop_word_ratio_high',
                'lf_title_sentence_count_high',
                'lf_title_word_is_negative',
                'lf_title_word_is_positive',
                'lf_title_word_not_negative',
                'lf_title_word_not_positive',
                'lf_content_avg_word_length_no_stop_words_low',
                'lf_content_avg_word_length_low',
                'lf_content_avg_words_per_sentence_high',
                'lf_title_proper_nouns_count_high',
                'lf_content_verb_ratio_low',
                'lf_content_avg_word_length_high',
                'lf_title_word_count_low',
                'lf_title_proper_nouns_ratio_high',
                'lf_content_exclamation_ratio_low',
                'lf_title_nouns_count_high',
                'lf_content_exclamation_count_fake',
                'lf_content_avg_word_length_medium_low',
                'lf_title_no_exclamation_marks'
                ]
                
NUMERICAL_COLS = ['id',
                  'published_utc',
                  'collection_utc',
                  'content_exclamation_count',
                  'title_exclamation_count',
                  'content_word_count',
                  'title_word_count',
                  'content_word_count_with_punctuation',
                  'title_word_count_with_punctuation',
                  'content_sentence_count',
                  'title_sentence_count',
                  'content_capital_word_count',
                  'title_capital_word_count',
                  'content_stop_word_count',
                  'title_stop_word_count',
                  'content_stop_word_ratio',
                  'title_stop_word_ratio',
                  'content_words_per_sentence',
                  'content_quote_marks_count',
                  'content_ttr_score',
                  'title_ttr_score',
                  'title_nouns_count',
                  'title_proper_nouns_count',
                  'content_avg_word_length',
                  'title_avg_word_length',
                  'content_avg_word_length_no_stop_words',
                  'title_avg_word_length_no_stop_words',
                  'content_url_count',
                  'content_verb_ratio',
                  'content_past_tense_verb_ratio',
                  'content_past_tense_verb_ratio_of_all_verbs',
                  'content_adjective_ratio',
                  'content_adverb_ratio',
                  'title_verb_ratio',
                  'title_past_tense_verb_ratio',
                  'title_past_tense_verb_ratio_of_all_verbs',
                  'title_adjective_ratio',
                  'title_adverb_ratio',
                  'content_capital_word_ratio',
                  'title_capital_word_ratio',
                  'content_personal_pronouns_count',
                  'content_personal_pronouns_ratio',
                  'content_quote_marks_ratio',
                  'title_nouns_ratio',
                  'title_proper_nouns_ratio',
                  'content_exclamation_ratio',
                  'title_exclamation_ratio',
                  'content_sentiment_word_sub',
                  'content_sentiment_word_pos',
                  'content_sentiment_word_neg',
                  'title_sentiment_word_sub',
                  'title_sentiment_word_pos',
                  'title_sentiment_word_neg',
                  'content_sentiment_sentence_sub',
                  'content_sentiment_sentence_pos',
                  'content_sentiment_sentence_neg',
                  'title_sentiment_sentence_sub',
                  'title_sentiment_sentence_pos',
                  'title_sentiment_sentence_neg',
                  'content_sentiment_text_sub',
                  'content_sentiment_text_pos',
                  'content_sentiment_text_neg',
                  'title_sentiment_text_sub',
                  'title_sentiment_text_pos',
                  'title_sentiment_text_neg',
                  'title_swn_pos_score',
                  'title_swn_neg_score',
                  'title_swn_obj_score',
                  'content_swn_pos_score',
                  'content_swn_neg_score',
                  'content_swn_obj_score'
                  ]

TEST_NUMERICAL_COLS = ['id',
                  'content_exclamation_count',
                  'title_exclamation_count',
                  'content_word_count',
                  'title_word_count',
                  'content_word_count_with_punctuation',
                  'title_word_count_with_punctuation',
                  'content_sentence_count',
                  'title_sentence_count',
                  'content_capital_word_count',
                  'title_capital_word_count',
                  'content_stop_word_count',
                  'title_stop_word_count',
                  'content_stop_word_ratio',
                  'title_stop_word_ratio',
                  'content_words_per_sentence',
                  'content_quote_marks_count',
                  'content_ttr_score',
                  'title_ttr_score',
                  'title_nouns_count',
                  'title_proper_nouns_count',
                  'content_avg_word_length',
                  'title_avg_word_length',
                  'content_avg_word_length_no_stop_words',
                  'title_avg_word_length_no_stop_words',
                  'content_url_count',
                  'content_verb_ratio',
                  'content_past_tense_verb_ratio',
                  'content_past_tense_verb_ratio_of_all_verbs',
                  'content_adjective_ratio',
                  'content_adverb_ratio',
                  'title_verb_ratio',
                  'title_past_tense_verb_ratio',
                  'title_past_tense_verb_ratio_of_all_verbs',
                  'title_adjective_ratio',
                  'title_adverb_ratio',
                  'content_capital_word_ratio',
                  'title_capital_word_ratio',
                  'content_personal_pronouns_count',
                  'content_personal_pronouns_ratio',
                  'content_quote_marks_ratio',
                  'title_nouns_ratio',
                  'title_proper_nouns_ratio',
                  'content_exclamation_ratio',
                  'title_exclamation_ratio',
                  'content_sentiment_word_sub',
                  'content_sentiment_word_pos',
                  'content_sentiment_word_neg',
                  'title_sentiment_word_sub',
                  'title_sentiment_word_pos',
                  'title_sentiment_word_neg',
                  'content_sentiment_sentence_sub',
                  'content_sentiment_sentence_pos',
                  'content_sentiment_sentence_neg',
                  'title_sentiment_sentence_sub',
                  'title_sentiment_sentence_pos',
                  'title_sentiment_sentence_neg',
                  'content_sentiment_text_sub',
                  'content_sentiment_text_pos',
                  'content_sentiment_text_neg',
                  'title_sentiment_text_sub',
                  'title_sentiment_text_pos',
                  'title_sentiment_text_neg',
                  'title_swn_pos_score',
                  'title_swn_neg_score',
                  'title_swn_obj_score',
                  'content_swn_pos_score',
                  'content_swn_neg_score',
                  'content_swn_obj_score'
                  ]

LIST_TYPE_COLS = ['content_tokenized_words',
                  'content_tokenized_sentences',
                  'content_tokenized_words_lowercase',
                  'content_tokenized_words_with_punctuation',
                  'content_words_no_stopwords',
                  'title_tokenized_words',
                  'title_tokenized_sentences',
                  'title_tokenized_words_lowercase',
                  'title_tokenized_words_with_punctuation',
                  'title_words_no_stopwords',
                  'content_pos_tags',
                  'title_pos_tags',
                  'title_pos_tags_lowercase',
                  'title_count_pos',
                  'content_urls'
                  ]

TEXT_COLS = ['id', 'content_lemmatized_lowercase_no_stopwords', 'title_lemmatized_lowercase_no_stopwords', 'content', 'title']

LEMMATIZED_COLS = ['id', 'content_lemmatized_lowercase_no_stopwords', 'title_lemmatized_lowercase_no_stopwords']
RAW_TEXT_COLS = ['id', 'content', 'title']
