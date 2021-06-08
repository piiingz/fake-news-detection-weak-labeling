import pandas as pd
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def calc_score(doc):
    pos_score = 0
    neg_score = 0
    obj_score = 0
    word_count = len(doc)
    for word in doc:
        s = list(swn.senti_synsets(word[0], word[1]))
        if not s:
            continue
        pos_score += s[0].pos_score()
        neg_score += s[0].neg_score()
        obj_score += s[0].obj_score()
    if word_count:
        pos, neg, obj = pos_score/word_count, neg_score/word_count, obj_score/word_count
    else:
        pos, neg, obj = 0, 0, 0
    return pd.Series([pos, neg, obj])


def swn_pos_tag(x):
    swn_t = list(map(lambda word: (word[0], get_wordnet_pos(word[1])), x))
    return swn_t


def apply_swn_score(subset: pd.DataFrame):
    subset['content_swn_pos_tags'] = subset['content_pos_tags_no_stopwords'].apply(swn_pos_tag)
    subset['title_swn_pos_tags'] = subset['title_pos_tags_no_stopwords'].apply(swn_pos_tag)

    subset[['title_swn_pos_score', 'title_swn_neg_score', 'title_swn_obj_score']] = subset[
        'title_swn_pos_tags'].apply(calc_score)
    subset[['content_swn_pos_score', 'content_swn_neg_score', 'content_swn_obj_score']] = subset[
        'content_swn_pos_tags'].apply(calc_score)
    return subset
