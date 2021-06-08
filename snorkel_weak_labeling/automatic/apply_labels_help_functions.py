# Libraries
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# LFs
import snorkel_weak_labeling.manual.complexity_labeling as complexity
import snorkel_weak_labeling.manual.pos_labeling as pos
import snorkel_weak_labeling.manual.sentiment_labeling as sentiment
import snorkel_weak_labeling.manual.stylistic_labeling as stylistic
from snorkel_weak_labeling.automatic.automated_lf_generation import generate_lfs_automatically

# Paths
from handle_datasets.paths import LF_CONFUSION_MATRIX_PATH


def get_lf_names():
    stylistic_lf_names = [x for x in dir(stylistic) if 'lf' in x]
    sentiment_lf_names = [x for x in dir(sentiment) if 'lf' in x]
    complexity_lf_names = [x for x in dir(complexity) if 'lf' in x]
    pos_lf_names = [x for x in dir(pos) if 'lf' in x]
    return stylistic_lf_names, sentiment_lf_names, complexity_lf_names, pos_lf_names


def get_all_lfs(DIFF_REL_THRESH, is_automated: bool = False):
    if is_automated:
        return generate_lfs_automatically(DIFF_REL_THRESH)
    stylistic_lf_names, sentiment_lf_names, complexity_lf_names, pos_lf_names = get_lf_names()
    lfs = []
    lfs.extend([stylistic.__dict__[x] for x in stylistic_lf_names])
    lfs.extend([sentiment.__dict__[x] for x in sentiment_lf_names])
    lfs.extend([complexity.__dict__[x] for x in complexity_lf_names])
    lfs.extend([pos.__dict__[x] for x in pos_lf_names])
    return lfs


def get_top_lfs(DIFF_REL_THRESH, nr_of_lfs: int, summary: pd.DataFrame, is_automated: bool = False):
    sorted_summary = summary.sort_values("Emp. Acc.", ascending=False)
    lf_names = sorted_summary.iloc[:nr_of_lfs, 0]
    return find_matching_lfs(DIFF_REL_THRESH, lf_names, is_automated=is_automated)


def get_acc_lfs(DIFF_REL_THRESH, acc: float, summary: pd.DataFrame, is_automated: bool = False):
    lf_names = summary.loc[summary["Emp. Acc."] >= acc].iloc[:, 0]
    return find_matching_lfs(DIFF_REL_THRESH, lf_names, is_automated=is_automated)


def find_matching_lfs(DIFF_REL_THRESH, lf_names: list, is_automated: bool = False):
    lfs = get_all_lfs(DIFF_REL_THRESH, is_automated)
    return [x for x in lfs if x.name in lf_names]


def write_results(path, gm_scores, mv_scores, gm_subset, mv_subset, analysis, original_subset):
    gm_fake_labels = gm_subset[gm_subset.gm_weak_label == 1].shape[0]
    gm_real_labels = gm_subset[gm_subset.gm_weak_label == 0].shape[0]
    gm_actual_fake = gm_subset[gm_subset.label == 1].shape[0]
    gm_actual_real = gm_subset[gm_subset.label == 0].shape[0]
    gm_rows = gm_subset.shape[0]

    mv_fake_labels = mv_subset[mv_subset.mv_weak_label == 1].shape[0]
    mv_real_labels = mv_subset[mv_subset.mv_weak_label == 0].shape[0]
    mv_actual_fake = mv_subset[mv_subset.label == 1].shape[0]
    mv_actual_real = mv_subset[mv_subset.label == 0].shape[0]
    mv_rows = mv_subset.shape[0]

    subset_original_size = original_subset.shape[0]
    subset_fake_size = original_subset[original_subset.label == 1].shape[0]
    subset_real_size = original_subset[original_subset.label == 0].shape[0]

    with open(path, 'w') as file:
        file.write("------ GENERATIVE MODEL RESULTS ------")
        file.write("\nGenerative model scores: " + json.dumps(gm_scores))
        file.write("\n% FAKE weak labels: {}".format(str(gm_fake_labels/gm_rows)))
        file.write("\n% REAL weak labels: {}".format(str(gm_real_labels/gm_rows)))
        file.write("\n# FAKE weak labels: {}/{}".format(str(gm_fake_labels), str(gm_rows)))
        file.write("\n# REAL weak labels: {}/{}".format(str(gm_real_labels), str(gm_rows)))

        file.write("\n\n% actual FAKE labels: {}".format(str(gm_actual_fake/gm_rows)))
        file.write("\n% actual REAL labels: {}".format(str(gm_actual_real/gm_rows)))
        file.write("\n# actual FAKE labels: {}/{}".format(str(gm_actual_fake), str(gm_rows)))
        file.write("\n# actual REAL labels: {}/{}".format(str(gm_actual_real), str(gm_rows)))

        file.write("\n\nCoverage of all instances: {}".format(str(gm_rows/subset_original_size)))
        file.write("\nCoverage of fake instances: {}".format(str(gm_actual_fake/subset_fake_size)))
        file.write("\nCoverage of real instances: {}".format(str(gm_actual_real/subset_real_size)))

        file.write("\n\n------ MAJORITY VOTE RESULTS ------")
        file.write("\nMajoirty vote scores: " + json.dumps(mv_scores))
        file.write("\n% FAKE weak labels: {}".format(str(mv_fake_labels/mv_rows)))
        file.write("\n% REAL weak labels: {}".format(str(mv_real_labels/mv_rows)))
        file.write("\n# FAKE weak labels: {}/{}".format(str(mv_fake_labels), str(mv_rows)))
        file.write("\n# REAL weak labels: {}/{}".format(str(mv_real_labels), str(mv_rows)))

        file.write("\n\n% actual FAKE labels: {}".format(str(mv_actual_fake/mv_rows)))
        file.write("\n% actual REAL labels: {}".format(str(mv_actual_real/mv_rows)))
        file.write("\n# actual FAKE labels: {}/{}".format(str(mv_actual_fake), str(mv_rows)))
        file.write("\n# actual REAL labels: {}/{}".format(str(mv_actual_real), str(mv_rows)))

        file.write("\n\nCoverage of all instances: {}".format(str(mv_rows / subset_original_size)))
        file.write("\nCoverage of fake instances: {}".format(str(mv_actual_fake/subset_fake_size)))
        file.write("\nCoverage of real instances: {}".format(str(mv_actual_real/subset_real_size)))

        file.write("\n\n --- LF Coverage --- \n")
        file.write('\nLabel coverage: ' + str(analysis.label_coverage()))
        file.write('\nLabel overlap : ' + str(analysis.label_overlap()))
        file.write('\nLabel conflict: ' + str(analysis.label_conflict()))


def plot_confusion_matrix(filtered_train: pd.DataFrame, name, is_majority_vote: bool = False):
    y_actual = filtered_train['label']
    if is_majority_vote:
        y_weak = filtered_train['mv_weak_label']
    else:
        y_weak = filtered_train['gm_weak_label']

    plt.figure(figsize=(8, 6))

    cf_matrix = confusion_matrix(y_actual, y_weak)
    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, cmap='Blues')
    accuracy = np.trace(cf_matrix) / np.sum(cf_matrix).astype('float')
    misclass = 1 - accuracy

    plt.title('Confusion matrix')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.ylabel('True label')
    plt.savefig('{}{}.png'.format(LF_CONFUSION_MATRIX_PATH, name))
    plt.close()


def create_postfix(DIFF_REL_THRESH,
                   is_top: bool = False,
                   is_acc: bool = False,
                   is_test: bool = False,
                   is_automated: bool = False):
    postfix = ''
    if is_top:
        postfix += '_top'
    elif is_acc:
        postfix += '_acc'
    else:
        postfix += '_all'
    if is_automated:
        postfix += '_auto' + str(DIFF_REL_THRESH)
    if is_test:
        postfix += '_test'
    else:
        postfix += '_train'
    return postfix
