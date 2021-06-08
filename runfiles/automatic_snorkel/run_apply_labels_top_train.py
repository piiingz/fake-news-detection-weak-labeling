from runfiles.automatic_snorkel.apply_labels_func import run_apply_labels
from config import AUTOMATED_LFS


if AUTOMATED_LFS:
    for i in range(1, 10):
        run_apply_labels(i/100, is_top=True)

    for i in range(1, 10):
        run_apply_labels(i/10, is_top=True)

else:
    run_apply_labels(0, is_top=True)