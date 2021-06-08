import pandas as pd

from config import NUMERICAL_COLS
from snorkel_weak_labeling.automatic.apply_labels import apply_labels, plot_confusion_matrix
from handle_datasets.old_load_subsets import get_test_subset

pd.set_option('display.expand_frame_repr', False, 'display.max_rows', None)

""" 
Requires the creation of folders:
    - data/weak_labeling/
    - data/weak_labeling/analysis/
    - data/weak_labeling/confusion_matrix/
"""

# train_subset = get_train_balanced_subset(usecols=['title', 'content'] + NUMERICAL_COLS)
# filtered_train, file_postfix = apply_labels(train_subset)
# plot_confusion_matrix(filtered_train, file_postfix)

test_subset = get_test_subset(usecols=['title', 'content'] + NUMERICAL_COLS)
filtered_test, majority_filtered_subset, file_postfix_test = apply_labels(test_subset, is_test=True)
plot_confusion_matrix(filtered_test, file_postfix_test)
plot_confusion_matrix(majority_filtered_subset, file_postfix_test + "_majority_vote", is_majority_vote=True)
