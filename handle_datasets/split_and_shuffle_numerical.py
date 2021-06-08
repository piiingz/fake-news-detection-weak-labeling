import pandas as pd
from sklearn.model_selection import train_test_split

from config import FAKE, REAL, TEST_SIZE, RANDOM_SEED, LEMMATIZED_COLS, RAW_TEXT_COLS
from handle_datasets.load_datasets import load_dataset_6_pkl, load_dataset_3_complete_text
from handle_datasets.save_datasets import save_numerical_test_train, save_numerical_lemma_test_train, \
    save_numerical_raw_test_train


def shuffle_and_split_numerical(df: pd.DataFrame):
    fake_total = df[df['label'] == FAKE]
    real_total = df[df['label'] == REAL]

    print("Fake total shape:", fake_total.shape)
    print("Real total shape:", real_total.shape)

    test_size = round(fake_total.shape[0] * TEST_SIZE)

    # Split into test and train
    fake_train, fake_test = train_test_split(fake_total, test_size=test_size, random_state=RANDOM_SEED, shuffle=True)
    real_train, real_test = train_test_split(real_total, test_size=test_size, random_state=RANDOM_SEED, shuffle=True)

    # Split real_train into one of equal size of fake_train, and one containing the rest
    real_train_balanced, real_train_excess = train_test_split(real_train, train_size=fake_train.shape[0], random_state=RANDOM_SEED, shuffle=True)

    # Concat fake and real into balanced train and test, and unbalanced train containing everything from train
    train_balanced = pd.concat([fake_train, real_train_balanced], ignore_index=True)
    test_balanced = pd.concat([fake_test, real_test], ignore_index=True)
    train_unbalanced = pd.concat([fake_train, real_train], ignore_index=True)

    # Shuffle
    train_balanced = train_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_balanced = test_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train_unbalanced = train_unbalanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Check that everything is correct
    print("Train balanced:", train_balanced.shape)
    print("Test balanced: ", test_balanced.shape)
    print("Train unbalanced: ", train_unbalanced.shape)

    print("Numerical cols: ", train_balanced.columns)

    return train_balanced, test_balanced, train_unbalanced


if __name__ == '__main__':

    # Save only numerical
    df = load_dataset_6_pkl()
    train_balanced, test_balanced, train_unbalanced = shuffle_and_split_numerical(df)
    save_numerical_test_train(train_balanced, test_balanced, train_unbalanced)

    # Load raw + lemmatized complete dataset
    text_features = load_dataset_3_complete_text()

    # Merge lemmatized with numerical and save
    train_balanced_lemma = pd.merge(train_balanced, text_features[LEMMATIZED_COLS], on='id')
    test_balanced_lemma = pd.merge(test_balanced, text_features[LEMMATIZED_COLS], on='id')
    train_unbalanced_lemma = pd.merge(train_unbalanced, text_features[LEMMATIZED_COLS], on='id')

    print("Train lemma balanced:", train_balanced_lemma.shape)
    print("Test lemma balanced: ", test_balanced_lemma.shape)
    print("Train lemma unbalanced: ", train_unbalanced_lemma.shape)

    save_numerical_lemma_test_train(train_balanced_lemma, test_balanced_lemma, train_unbalanced_lemma)

    # Merge raw text with numerical and save
    train_balanced_raw = pd.merge(train_balanced, text_features[RAW_TEXT_COLS], on='id')
    test_balanced_raw = pd.merge(test_balanced, text_features[RAW_TEXT_COLS], on='id')
    train_unbalanced_raw = pd.merge(train_unbalanced, text_features[RAW_TEXT_COLS], on='id')

    print("Train raw balanced:", train_balanced_raw.shape)
    print("Test raw balanced: ", test_balanced_raw.shape)
    print("Train raw unbalanced: ", train_unbalanced_raw.shape)

    save_numerical_raw_test_train(train_balanced_raw, test_balanced_raw, train_unbalanced_raw)
