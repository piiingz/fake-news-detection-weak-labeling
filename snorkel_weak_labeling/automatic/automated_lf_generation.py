from snorkel_weak_labeling.automatic.threshold_search import *
from config import FAKE, REAL, ABSTAIN
from snorkel.labeling import LabelingFunction


def make_upper_lf(real, fake, feature):
    def upper_real(x):
        return REAL if x[feature] > fake else ABSTAIN

    def upper_fake(x):
        return FAKE if x[feature] > real else ABSTAIN

    if real > fake:
        return LabelingFunction(
            name=f"lf_{feature}_upper",
            f=upper_real,
        )
    if fake > real:
        return LabelingFunction(
            name=f"lf_{feature}_upper",
            f=upper_fake,
        )


def make_lower_lf(real, fake, feature):
    def lower_fake(x):
        return FAKE if x[feature] < real else ABSTAIN

    def lower_real(x):
        return REAL if x[feature] < fake else ABSTAIN

    if real > fake:
        return LabelingFunction(
            name=f"lf_{feature}_lower",
            f=lower_fake,
        )
    if fake > real:
        return LabelingFunction(
            name=f"lf_{feature}_lower",
            f=lower_real,
        )


def generate_lfs_automatically(DIFF_REL_THRESH):
    thresholds = search(DIFF_REL_THRESH)
    lfs = []
    for i, row in thresholds.iterrows():
        if not pd.isnull(row['real_upper']) and not pd.isnull(row['fake_upper']):
            lf_u = make_upper_lf(row['real_upper'], row['fake_upper'], row['feature'])
            if lf_u:
                lfs.append(lf_u)
        if not pd.isnull(row['real_lower']) and not pd.isnull(row['fake_lower']):
            lf_l = make_lower_lf(row['real_lower'], row['fake_lower'], row['feature'])
            if lf_l:
                lfs.append(lf_l)
    return lfs
