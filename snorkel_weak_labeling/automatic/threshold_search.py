import pandas as pd
from handle_datasets.paths import FAKE_SUBSET_DESCRIBE_PATH, REAL_SUBSET_DESCRIBE_PATH, THRESHOLD_PATH
from config import UPPER_QUANTILES, LOWER_QUANTILES


def diff_above_thresh(real, fake, mean, max, min, DIFF_REL_THRESH):
    if mean == 0:
        mean = 1
    scaled_real = 1/(max-min)*(real-min)
    scaled_fake = 1/(max-min)*(fake-min)
    scaled_mean = 1/(max-min)*(mean-min)
    if abs(scaled_real-scaled_fake)/scaled_mean >= DIFF_REL_THRESH:
        return True
    return False


def search(DIFF_REL_THRESH):
    real_description = pd.read_csv(REAL_SUBSET_DESCRIBE_PATH, index_col=[0])
    fake_description = pd.read_csv(FAKE_SUBSET_DESCRIBE_PATH, index_col=[0])
    real_description.drop(["label", "id"], axis=1, inplace=True)
    fake_description.drop(["label", "id"], axis=1, inplace=True)
    thresholds = pd.DataFrame(columns=['feature', 'fake_upper', 'real_upper', 'fake_lower', 'real_lower', 'upper_quantile', 'lower_quantile'])

    for feature in fake_description.columns:
        fake_upper, fake_lower, real_upper, real_lower, upper_quantile, lower_quantile = None, None, None, None, None, None
        q_real = real_description[feature]
        q_fake = fake_description[feature]

        # Upper
        for q in UPPER_QUANTILES:
            if diff_above_thresh(q_real[q], q_fake[q], q_real['mean'], q_real['max'], q_real['min'], DIFF_REL_THRESH):
                real_upper = q_real[q]
                fake_upper = q_fake[q]
                upper_quantile = q
                break

        for q in LOWER_QUANTILES:
            if diff_above_thresh(q_real[q], q_fake[q], q_real['mean'], q_real['max'], q_real['min'], DIFF_REL_THRESH):
                real_lower = q_real[q]
                fake_lower = q_fake[q]
                lower_quantile = q
                break

        thresholds.loc[len(thresholds)] = [feature, fake_upper, real_upper, fake_lower, real_lower, upper_quantile, lower_quantile]

    thresholds.to_csv(THRESHOLD_PATH + str(DIFF_REL_THRESH) + '.csv')
    return thresholds
