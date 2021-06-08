import numpy as np
import pandas as pd
import pickle
from snuba_weak_labeling.program_synthesis.heuristic_generator import HeuristicGenerator
from snuba_weak_labeling.data.loader import load_labeled, load_unlabeled, load_dataframe_local
import warnings
from handle_datasets.paths import SNUBA_LABELS_UNLABELED_PATH, SNUBA_LABELS_LABELED_PATH, SNUBA_TEST_LABELS_UNLABELED_PATH, SNUBA_TEST_LABELS_LABELED_PATH
from handle_datasets.paths import SNUBA_HEURISTICS_STATS_PATH, SNUBA_HEURISTICS_PATH, SNUBA_GEN_MODEL_PATH, SNUBA_BETA_OPT_PATH, SNUBA_FEAT_COMBOS_PATH
from handle_datasets.paths import SNUBA_TEST_HEURISTICS_STATS_PATH, SNUBA_TEST_HEURISTICS_PATH, SNUBA_TEST_GEN_MODEL_PATH, SNUBA_TEST_BETA_OPT_PATH, SNUBA_TEST_FEAT_COMBOS_PATH
from handle_datasets.paths import SNUBA_SCORES_PATH, SNUBA_SCORES_PATH_SNORKEL_GM, SNUBA_SCORES_PATH_SNORKEL_MV
from config import IS_LOCAL


"""
Code source for Snuba/reef: https://github.com/HazyResearch/reef
"""


def snubify_path(path: str, cardinality: int, model_type: str, txt: bool = False):
    p = path + model_type + '_' + str(cardinality)
    if txt:
        return p + '.txt'
    return p + '.csv'


def write_results(train_acc,
                  train_cov,
                  train_f1,
                  snuba_train_labels,
                  ground_train_labels,
                  val_acc,
                  val_cov,
                  val_f1,
                  snuba_val_labels,
                  ground_val_labels,
                  is_test,
                  has_snorkel,
                  use_mv,
                  cardinality,
                  model_type):

    if has_snorkel:
        if use_mv:
            path = snubify_path(SNUBA_SCORES_PATH_SNORKEL_MV, cardinality, model_type, txt=True)
        else:
            path = snubify_path(SNUBA_SCORES_PATH_SNORKEL_GM, cardinality, model_type, txt=True)
    else:
        path = snubify_path(SNUBA_SCORES_PATH, cardinality, model_type, txt=True)

    if is_test:
        path = path[:-4] + '_test.txt'


    train_non_abstain = snuba_train_labels[snuba_train_labels != 0.5]
    val_non_abstain = snuba_val_labels[snuba_val_labels != 0.5]

    ground_train_non_abstain = ground_train_labels[snuba_train_labels.index]
    ground_val_non_abstain = ground_val_labels[snuba_val_labels.index]

    train_actual_fake = np.shape(np.where(ground_train_non_abstain[ground_train_non_abstain == 1]))[1]
    train_actual_real = np.shape(np.where(ground_train_non_abstain[ground_train_non_abstain == -1]))[1]

    val_actual_fake = np.shape(np.where(ground_val_non_abstain[ground_val_non_abstain == 1]))[1]
    val_actual_real = np.shape(np.where(ground_val_non_abstain[ground_val_non_abstain == -1]))[1]

    with open(path, 'w') as f:
        f.write("\nTrain Accuracy: {}".format(train_acc))
        f.write("\nTrain F1-score: {}".format(train_f1))
        f.write("\nTrain Coverage: {}".format(train_cov))
        f.write("\nTrain Coverage Fake:  {}".format(train_actual_fake/train_non_abstain.shape[0]))
        f.write("\nTrain Coverage Real:  {}".format(train_actual_real/train_non_abstain.shape[0]))

        f.write("\n\nValidation Accuracy: {}".format(val_acc))
        f.write("\nValidation F1-score: {}".format(val_f1))
        f.write("\nValidation Coverage: {}".format(val_cov))
        f.write("\nValidation Coverage Fake:  {}".format(val_actual_fake/val_non_abstain.shape[0]))
        f.write("\nValidation Coverage Real:  {}".format(val_actual_real/val_non_abstain.shape[0]))
    

def main(cardinality, model_type, is_test: bool = False, has_snorkel: bool = False, use_mv: bool = False):
    """
    Snuba generates heuristics in an iterative manner, with each iteration consisting of the following steps:

    1. Synthesize Heuristics
    2. Prune Heuristics
    3. Verify Heuristics
    """

    warnings.filterwarnings("ignore")

    # Load data
    if IS_LOCAL:
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
        train_ground, val_ground, test_ground, columns, train_id, val_id = load_dataframe_local(is_test)
    elif is_test:
        train_primitive_matrix, val_primitive_matrix, \
        train_ground, val_ground, columns, train_id, val_id = load_labeled()
    else:
        train_primitive_matrix, val_primitive_matrix, \
        train_ground, val_ground, columns, train_id, val_id = load_unlabeled()

    # Print the shape of training, val and test sets
    print(train_primitive_matrix.shape, val_primitive_matrix.shape, \
    train_ground.shape, val_ground.shape, train_id.shape, val_id.shape)

    validation_accuracy = []
    training_accuracy = []

    validation_coverage = []
    training_coverage = []

    validation_f1 = []
    training_f1 = []

    validation_marginals = []
    training_marginals = []

    idx = None

    hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix,
                            val_ground, train_ground,
                            b=0.5, has_snorkel=has_snorkel, use_mv=use_mv)

    for i in range(3, 26):
        
        print("Running iteration: ", str(i - 2))

        # Repeat synthesize-prune-verify at each iterations
        if i == 3:
            hg.run_synthesizer(max_cardinality=cardinality, idx=idx, keep=3, model=model_type)
        else:
            hg.run_synthesizer(max_cardinality=cardinality, idx=idx, keep=1, model=model_type)
        hg.run_verifier()

        # Save evaluation metrics
        va, ta, vc, tc, vf1, tf1 = hg.evaluate()
        validation_accuracy.append(va)
        training_accuracy.append(ta)
        validation_marginals.append(hg.vf.val_marginals)
        training_marginals.append(hg.vf.train_marginals)
        validation_coverage.append(vc)
        training_coverage.append(tc)
        validation_f1.append(vf1)
        training_f1.append(tf1)

        # Find low confidence datapoints in the labeled set
        hg.find_feedback()
        idx = hg.feedback_idx

        # Stop the iterative process when no low confidence labels
        if idx == []:
            break
    
    snuba_train_labels = pd.DataFrame({
        "id": train_id,
        "prob_label": training_marginals[-1],
        "weak_label": np.sign(2 * (training_marginals[-1] - 0.5)),
        "ground_label": train_ground
        }
    )
    snuba_val_labels = pd.DataFrame({
        "id": val_id,
        "val_prob_label": validation_marginals[-1],
        "val_weak_label": np.sign(2 * (validation_marginals[-1] - 0.5)),
        "val_ground_label": val_ground
        }
    )

    h_stats = hg.heuristic_stats()

    write_results(
        training_accuracy[-1], 
        training_coverage[-1],
        training_f1[-1],
        snuba_train_labels,
        train_ground,
        validation_accuracy[-1],
        validation_coverage[-1],
        validation_f1[-1],
        snuba_val_labels,
        val_ground,
        is_test,
        has_snorkel,
        use_mv,
        cardinality,
        model_type
        )

    # Save labels to csv
    if is_test:
        snuba_train_labels.to_csv(snubify_path(SNUBA_TEST_LABELS_UNLABELED_PATH, cardinality, model_type))
        snuba_val_labels.to_csv(snubify_path(SNUBA_TEST_LABELS_LABELED_PATH, cardinality, model_type))

        with open(snubify_path(SNUBA_TEST_HEURISTICS_PATH, cardinality, model_type), 'wb') as f:
            pickle.dump(hg.hf, f)
    
        with open(snubify_path(SNUBA_TEST_GEN_MODEL_PATH, cardinality, model_type), 'wb') as f:
            pickle.dump(hg.vf.gen_model, f)

        with open(snubify_path(SNUBA_TEST_BETA_OPT_PATH, cardinality, model_type), 'wb') as f:
            pickle.dump(hg.beta_opt, f)

        with open(snubify_path(SNUBA_TEST_FEAT_COMBOS_PATH, cardinality, model_type), 'wb') as f:
            pickle.dump(hg.feat_combos, f)
        
        h_stats.to_csv(snubify_path(SNUBA_TEST_HEURISTICS_STATS_PATH, cardinality, model_type))

    else:
        snuba_train_labels.to_csv(snubify_path(SNUBA_LABELS_UNLABELED_PATH, cardinality, model_type))
        snuba_val_labels.to_csv(snubify_path(SNUBA_LABELS_LABELED_PATH, cardinality, model_type))
        
        curr_path = snubify_path(SNUBA_HEURISTICS_PATH, cardinality, model_type)
        with open(curr_path, 'wb') as f:
            pickle.dump(hg.hf, f)

        curr_path = snubify_path(SNUBA_GEN_MODEL_PATH, cardinality, model_type)
        with open(curr_path, 'wb') as f:
            pickle.dump(hg.vf.gen_model, f)

        curr_path = snubify_path(SNUBA_BETA_OPT_PATH, cardinality, model_type)
        with open(curr_path, 'wb') as f:
            pickle.dump(hg.beta_opt, f)

        curr_path = snubify_path(SNUBA_FEAT_COMBOS_PATH, cardinality, model_type)
        with open(curr_path, 'wb') as f:
            pickle.dump(hg.feat_combos, f)

        h_stats.to_csv(snubify_path(SNUBA_HEURISTICS_STATS_PATH, cardinality, model_type))


if __name__ == '__main__':
    main()
