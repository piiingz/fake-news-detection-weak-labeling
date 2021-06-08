import sys
from config import IS_LOCAL, MODEL_TYPES
from snuba_weak_labeling.apply_labels import main


if IS_LOCAL:
    main(2, 'lr')
else:
    is_test = bool(int(sys.argv[1]))        # = var1 is either 0 (False) or 1 (True)
    has_snorkel = bool(int(sys.argv[2]))    # = var2 is either 0 (False) or 1 (True)
    use_mv = bool(int(sys.argv[3]))         # = var3 is either 0 (False) or 1 (True)
    cardinality = int(sys.argv[4])          # = var4 is either 1, 2, 3 or 4
    model_type = str(sys.argv[5])           # = var5 is either 'dt', 'lr' or 'nn'
    main(cardinality, model_type, is_test, has_snorkel, use_mv)

