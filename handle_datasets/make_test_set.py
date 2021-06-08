from handle_datasets.old_load_subsets import *
from handle_datasets.old_paths import *
from config import *
from sklearn.model_selection import train_test_split


def shuffle_and_split_datasets(fake_total: pd.DataFrame, real_total: pd.DataFrame):

    # Add labels
    fake_total['label'] = FAKE
    real_total['label'] = REAL

    test_size = round(len(fake_total.index) * TEST_SIZE)

    # Split into test and train
    fake_train, fake_test = train_test_split(fake_total, test_size=test_size, random_state=RANDOM_SEED, shuffle=True)
    real_train, real_test = train_test_split(real_total, test_size=test_size, random_state=RANDOM_SEED, shuffle=True)

    # Split real_train into one of equal size of fake_train, and one containing the rest
    real_train_balanced, real_train_excess = train_test_split(real_train, train_size=len(fake_train.index),
                                                              random_state=RANDOM_SEED, shuffle=True)

    # Concat fake and real into balanced train and test, and unbalanced train containing everything from train
    train_balanced = pd.concat([fake_train, real_train_balanced], ignore_index=True)
    test_balanced = pd.concat([fake_test, real_test], ignore_index=True)
    train_unbalanced = pd.concat([fake_train, real_train], ignore_index=True)

    # Shuffle
    train_balanced = train_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_balanced = test_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train_unbalanced = train_unbalanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Drop column unnamed if exists
    train_balanced = train_balanced.drop(["Unnamed: 0"], axis=1, errors='ignore')
    test_balanced = test_balanced.drop(["Unnamed: 0"], axis=1, errors='ignore')
    train_unbalanced = train_unbalanced.drop(["Unnamed: 0"], axis=1, errors='ignore')

    # Check that everything is correct
    print("Train balanced:", train_balanced.shape)
    print("Test balanced: ", test_balanced.shape)
    print("Train unbalanced: ", train_unbalanced.shape)

    return train_balanced, test_balanced, train_unbalanced


def save_test_and_train_after_split():

    fake_total, real_total = assemble_datasets_after_split()

    train_balanced, test_balanced, train_unbalanced = shuffle_and_split_datasets(fake_total, real_total)

    save_numerical_features_to_csv(train_unbalanced, TRAIN_TOTAL_NUMERICAL_COLS_PATH)
    save_numerical_features_to_csv(train_balanced, TRAIN_BALANCED_NUMERICAL_COLS_PATH)
    save_numerical_features_to_csv(test_balanced, TEST_NUMERICAL_COLS_PATH)

    # Save to CSV
    train_balanced.to_csv(TRAIN_BALANCED_PREPROCESSED_PATH, encoding='utf-8')
    test_balanced.to_csv(TEST_PREPROCESSED_PATH, encoding='utf-8')
    train_unbalanced.to_csv(TRAIN_TOTAL_PREPROCESSED_PATH, encoding='utf-8')
