import pandas as pd
from handle_datasets.paths import FAKE_SUBSET_DESCRIBE_PATH, REAL_SUBSET_DESCRIBE_PATH


def describe_train_features(subset: pd.DataFrame):
    fake_features = subset.loc[subset.label == 1]
    real_features = subset.loc[subset.label == 0]

    print("Subset shape: ", subset.shape)
    print("Fake: ", fake_features.shape)
    print("Real: ", real_features.shape)

    percentiles = [.05, .10, .15, .20, .25, .30, .50, .70, .75, .80, .85, .90, .95]

    fake_description = fake_features.describe(percentiles=percentiles, datetime_is_numeric=True)
    real_description = real_features.describe(percentiles=percentiles, datetime_is_numeric=True)

    fake_description.to_csv(FAKE_SUBSET_DESCRIBE_PATH, encoding='utf8')
    real_description.to_csv(REAL_SUBSET_DESCRIBE_PATH, encoding='utf8')
