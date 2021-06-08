from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from handle_datasets.paths import LF_SUMMARY_PATH, LABEL_MODEL_SCORE_PATH, TRAIN_BALANCED_WEAK_LABELED_PATH
from snorkel_weak_labeling.automatic.apply_labels_help_functions import *


def evaluate_lfs(subset: pd.DataFrame, lfs: list, file_postfix: str = ''):
    summary = get_summary(subset, lfs)
    summary.to_csv(LF_SUMMARY_PATH + file_postfix + '.csv', encoding='utf-8')


def get_summary(subset: pd.DataFrame, lfs: list):

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=subset)

    # Analyse LFs
    analysis = LFAnalysis(L=L_train, lfs=lfs)
    summary = analysis.lf_summary(Y=subset['label'].to_numpy())
    return summary


def apply_labels(subset: pd.DataFrame,
                 lfs: list,
                 file_postfix: str = ''):

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=subset)
    analysis = LFAnalysis(L=L_train, lfs=lfs)

    generative_model = LabelModel(cardinality=2, verbose=True)
    generative_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
    gm_scores = generative_model.score(L=L_train, Y=subset['label'], metrics=['accuracy', 'f1'])

    majority_voter = MajorityLabelVoter()
    mv_scores = majority_voter.score(L=L_train, Y=subset['label'], metrics=['accuracy', 'f1'])

    gm_preds_train = generative_model.predict(L_train)
    mv_preds_train = majority_voter.predict(L_train)
    subset['gm_weak_label'] = gm_preds_train
    subset['mv_weak_label'] = mv_preds_train

    gm_filtered_subset = subset.loc[subset['gm_weak_label'] != -1]
    mv_filtered_subset = subset.loc[subset['mv_weak_label'] != -1]

    write_results(LABEL_MODEL_SCORE_PATH + file_postfix + '.txt',
                  gm_scores,
                  mv_scores,
                  gm_filtered_subset,
                  mv_filtered_subset,
                  analysis,
                  subset
                  )

    subset.to_csv(TRAIN_BALANCED_WEAK_LABELED_PATH + file_postfix + '.csv', encoding='utf-8')

    return gm_filtered_subset, mv_filtered_subset



