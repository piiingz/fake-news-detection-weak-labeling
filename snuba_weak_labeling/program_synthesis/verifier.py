import numpy as np
from scipy import sparse
from snuba_weak_labeling.program_synthesis.label_aggregator import LabelAggregator

"""
Code source for Snuba/reef: https://github.com/HazyResearch/reef
"""

def odds_to_prob(l):
  """
  This is the inverse logit function logit^{-1}:
    l       = \log\frac{p}{1-p}
    \exp(l) = \frac{p}{1-p}
    p       = \frac{\exp(l)}{1 + \exp(l)}
  """
  return np.exp(l) / (1.0 + np.exp(l))

class Verifier(object):
    """
    A class for the Snorkel Model Verifier
    """

    def __init__(self, L_train, L_val, val_ground, has_snorkel=False, use_mv=False):
        self.L_train = L_train.astype(int)
        self.L_val = L_val.astype(int)
        self.val_ground = val_ground

        self.has_snorkel = has_snorkel
        self.use_mv = use_mv

    def train_gen_model(self,deps=False,grid_search=False):
        """
        Calls appropriate generative model
        """
        if self.has_snorkel:
            if self.use_mv:
                from snorkel.labeling.model import MajorityLabelVoter
                gen_model = MajorityLabelVoter()
            else:
                from snorkel.labeling.model import LabelModel
                gen_model = LabelModel()
                gen_model.fit(L_train=self.L_train, n_epochs=500, log_freq=100, seed=123)
        else:
            gen_model = LabelAggregator()
            gen_model.train(self.L_train, rate=1e-3, mu=1e-6, verbose=False)
        self.gen_model = gen_model

    def assign_marginals(self):
        """
        Assigns probabilistic labels for train and val sets
        """
        if self.has_snorkel:
            self.train_marginals = self.gen_model.predict_proba(self.L_train)[:, 1]
            self.val_marginals = self.gen_model.predict_proba(self.L_val)[:, 1]
        else:
            self.train_marginals = self.gen_model.marginals(sparse.csr_matrix(self.L_train))
            self.val_marginals = self.gen_model.marginals(sparse.csr_matrix(self.L_val))
        #print 'Learned Accuracies: ', odds_to_prob(self.gen_model.w)

    def find_vague_points(self,gamma=0.1,b=0.5):
        """
        Find val set indices where marginals are within thresh of b
        """
        val_idx = np.where(np.abs(self.val_marginals-b) <= gamma)
        return val_idx[0]

    def find_incorrect_points(self,b=0.5):
        """ Find val set indices where marginals are incorrect """
        val_labels = 2*(self.val_marginals > b)-1
        val_idx = np.where(val_labels != self.val_ground)
        return val_idx[0]