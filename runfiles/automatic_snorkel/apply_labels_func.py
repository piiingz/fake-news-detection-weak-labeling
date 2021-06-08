import pandas as pd
from config import NR_OF_LFS, ACC_OF_LFS, AUTOMATED_LFS
from handle_datasets.load_datasets import load_dataset_6_train_balanced
from snorkel_weak_labeling.automatic.apply_labels import apply_labels, plot_confusion_matrix, evaluate_lfs, get_summary
from snorkel_weak_labeling.automatic.apply_labels_help_functions import get_all_lfs, get_top_lfs, get_acc_lfs, create_postfix


def run_apply_labels(DIFF_REL_THRESH,
                     is_top: bool = False,
                     is_acc: bool = False,
                     is_test: bool = False,
                     is_automated: bool = AUTOMATED_LFS):

    pd.set_option('display.expand_frame_repr', False, 'display.max_rows', None)
    file_postfix = create_postfix(DIFF_REL_THRESH, is_top=is_top, is_acc=is_acc, is_test=is_test, is_automated=is_automated)

    # LOADING THE DATA NEEDS TO BE CHANGED AFTER REFACTORING
    if is_test:
        # This must be changed after we create a test set
        subset = None
    else:
        subset = load_dataset_6_train_balanced()

    print(subset.columns)

    all_lfs = get_all_lfs(DIFF_REL_THRESH, is_automated)

    if is_top:
        summary = get_summary(subset, all_lfs)
        lfs = get_top_lfs(DIFF_REL_THRESH, NR_OF_LFS, summary, is_automated)
    elif is_acc:
        summary = get_summary(subset, all_lfs)
        lfs = get_acc_lfs(DIFF_REL_THRESH, ACC_OF_LFS, summary, is_automated)
    else:
        lfs = all_lfs

    evaluate_lfs(subset, lfs, file_postfix=file_postfix)
    gm_train, mv_train = apply_labels(subset, lfs, file_postfix)
    plot_confusion_matrix(gm_train, file_postfix)
    plot_confusion_matrix(mv_train, file_postfix)
