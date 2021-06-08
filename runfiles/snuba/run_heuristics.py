from snuba_weak_labeling.program_synthesis.heuristic_generator import HeuristicGenerator
from snuba_weak_labeling.program_synthesis.verifier import Verifier
from snuba_weak_labeling.data.loader import load_unlabeled
from snuba_weak_labeling.data.loader import load_dataframe_local
from handle_datasets.paths import SNUBA_GOAL_HEURISTICS_PATH, SNUBA_GOAL_GEN_MODEL_PATH, SNUBA_GOAL_BETA_OPT_PATH, \
    SNUBA_GOAL_FEAT_COMBOS_PATH, SNUBA_LABELS_UNLABELED_PATH
from config import GOAL_MAX_CARDINALITY, IS_LOCAL
import pickle
import numpy as np
import pandas as pd


def main(has_snorkel: bool = False):
    
    # Load data
    if IS_LOCAL:
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
        train_ground, val_ground, test_ground, columns = load_dataframe_local()
    else:
        train_primitive_matrix, val_primitive_matrix, train_ground, val_ground, columns = load_unlabeled()

    hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix,
                            val_ground, train_ground,
                            b=0.5, has_snorkel=has_snorkel)
    
    with open(SNUBA_GOAL_HEURISTICS_PATH, 'rb') as f:
        hf = pickle.load(f)
    
    with open(SNUBA_GOAL_GEN_MODEL_PATH, 'rb') as f:
        gen_model = pickle.load(f)

    with open(SNUBA_GOAL_BETA_OPT_PATH, 'rb') as f:
        beta_opt = pickle.load(f)

    with open(SNUBA_GOAL_FEAT_COMBOS_PATH, 'rb') as f:
        feat_combos = pickle.load(f)

    feat_cols = []
    for combo in feat_combos:
        temp = []
        for i in range(GOAL_MAX_CARDINALITY):
            temp.append(columns[combo[i]])
        feat_cols.append(temp)

    L_train = hg.apply_heuristics(hf, train_primitive_matrix, feat_combos, beta_opt)
    L_val = hg.apply_heuristics(hf, val_primitive_matrix, feat_combos, beta_opt)

    vf = Verifier(L_train, L_val, val_ground, has_snorkel=False)
    vf.gen_model = gen_model
    vf.assign_marginals()

    labels = pd.DataFrame({
        "prob_label": vf.training_marginals,
        "weak_label": np.sign(2 * (vf.train_marginals - 0.5)),
        "ground_label": train_ground
        }
    )

    labels.to_csv(SNUBA_LABELS_UNLABELED_PATH)


if __name__ == '__main__':
    main()