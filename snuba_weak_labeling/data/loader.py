import numpy as np
import sklearn.model_selection
import pandas as pd
from handle_datasets.load_datasets import load_dataset_6_train_balanced, load_dataset_6_test_balanced, load_dataset_test_preprocessed_pkl
from config import NUM_TEST, VAL_RATIO

"""

Code source for Snuba/reef: https://github.com/HazyResearch/reef

"""


def load_dataframe_local(is_test: bool = False, val_ratio: float = VAL_RATIO):
    df = pd.read_csv("data/all_preprocessed/train/train_total_numerical.csv")[:100]
    df['snuba_label'] = df['label'].apply(lambda x: -1 if x == 0 else 1)
    y = df.snuba_label
    X = df.drop(["label",
                 "snuba_label",
                 "title_swn_pos_score",
                 "title_swn_neg_score",
                 "title_swn_obj_score",
                 "content_swn_pos_score",
                 "content_swn_neg_score",
                 "content_swn_obj_score"], axis=1).fillna(0)

    np.random.seed(1234)
    num_sample = np.shape(X)[0]
    # num_test = 50

    # X_test = X.loc[0:num_test, :]
    X_train = X

    # y_test = y.loc[0:num_test]
    y_train = y

    # split dev/test
    X_tr, X_val, y_tr, y_val = \
        sklearn.model_selection.train_test_split(X_train, y_train, test_size=val_ratio)

    return np.array(X_tr), np.array(X_val), np.array(0), \
       np.array(y_tr), np.array(y_val), np.array(0), X.columns, X_tr.index, X_val.index
    

def load_dataframe(is_test: bool = False, val_ratio: float = VAL_RATIO):
    if is_test:
        df = load_dataset_6_test_balanced()
    else:
        df = load_dataset_6_train_balanced()
    df['snuba_label'] = df['label'].apply(lambda x: -1 if x == 0 else 1)
    df_id = df['id']
    y = df.snuba_label
    X = df.drop(["label", "snuba_label", 'published_utc', 'collection_utc', 'id'], errors='ignore', axis=1).fillna(0)

    np.random.seed(1234)
    num_sample = np.shape(X)[0]

    X_test = X.iloc[0:NUM_TEST, :]
    X_train = X.iloc[NUM_TEST:, :]

    y_test = y.iloc[0:NUM_TEST]
    y_train = y.iloc[NUM_TEST:]

    id_test = df_id[0:NUM_TEST]
    id_train = df_id[NUM_TEST:]

    # split dev/test
    X_tr, X_val, y_tr, y_val = \
        sklearn.model_selection.train_test_split(X_train, y_train, test_size=val_ratio)

    return np.array(X_tr), np.array(X_val), np.array(X_test), \
       np.array(y_tr), np.array(y_val), np.array(y_test), X.columns, id_train, id_test


def load_labeled():
    df = load_dataset_6_test_balanced()
    df['snuba_label'] = df['label'].apply(lambda x: -1 if x == 0 else 1)
    df_id = df['id']
    y = df.snuba_label
    X = df.drop(["label", "snuba_label", 'published_utc', 'collection_utc', 'id'], errors='ignore', axis=1).fillna(0)

    X_u = X.iloc[0:500, :]
    y_u = y.iloc[0:500]
    id_u = X.iloc[0:500]

    X_l = X
    y_l = y
    id_l = df_id
    
    return np.array(X_u), np.array(X_l), np.array(y_u), np.array(y_l), X.columns, id_u, id_l


def load_unlabeled():
    df_u = load_dataset_test_preprocessed_pkl()
    df_u['snuba_label'] = df_u['label'].apply(lambda x: -1 if x == 0 else 1)
    id_u = df_u['id']
    y_u = df_u.snuba_label
    X_u = df_u.drop(["label", "snuba_label", 'published_utc', 'collection_utc', 'id'], errors='ignore', axis=1).fillna(0)

    df_l = load_dataset_6_test_balanced()
    df_l['snuba_label'] = df_l['label'].apply(lambda x: -1 if x == 0 else 1)
    id_l = df_l['id']
    y_l = df_l.snuba_label
    X_l = df_l.drop(["label", "snuba_label", 'published_utc', 'collection_utc', 'id'], errors='ignore', axis=1).fillna(0)
    
    return np.array(X_u), np.array(X_l), np.array(y_u), np.array(y_l), X_l.columns, id_u, id_l


