import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    """Base class for model selection (strategy design pattern)."""

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states, X=None, lengths=None):
        """Train base model."""
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if X is None or lengths is None:
            X, lengths = self.X, self.lengths
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            if True or np.allclose(hmm_model.transmat_.sum(axis=1), 1.):
                return hmm_model
            else:
                print("failure on {} with {} states".format(self.this_word, num_states))
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """Select the model with value self.n_constant."""

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """Select the model with the lowest Bayesian Information Criterion(BIC) score.

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def score_bic(self, n_components, model):
        """Get BIC score."""
        log_likelihood = model.score(self.X, self.lengths)
        n_estimated_parameters = n_components ** 2  # See class docstring link.
        bic = -2 * log_likelihood + n_estimated_parameters * np.log(sum(self.lengths))
        return bic

    def select(self):
        """Select the best model for self.this_word based on BIC score
        for n between self.min_n_components and self.max_n_components.

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        optimal_score = float('inf')
        optimal_model = None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n_components)
            if model is not None:
                bic = self.score_bic(n_components, model)
                if bic < optimal_score:
                    optimal_score = bic
                    optimal_model = model
        return optimal_model


class SelectorDIC(ModelSelector):
    """Select best model based on Discriminative Information Criterion.

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    """

    def score_dic(self, model):
        """Get DIC score."""
        scores = []
        for word, (X, lengths) in self.hwords.items():
            if word != self.this_word:
                scores.append(model.score(X, lengths))
        return model.score(self.X, self.lengths) - np.mean(scores)

    def select(self):
        """Select the best model for self.this_word based on
        DIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        optimal_score = -float('inf')
        optimal_model = None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n_components)
            if model is not None:
                dic = self.score_dic(model)
                if dic > optimal_score:
                    optimal_score = dic
                    optimal_model = model
        return optimal_model


class SelectorCV(ModelSelector):
    """Select best model based on average log Likelihood of cross-validation folds."""

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        optimal_score = -float('inf')
        optimal_model = None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            cv_scores = []
            for train_indices, test_indices in KFold(n_splits=2).split(self.sequences):
                X_train, lengths_train = combine_sequences(train_indices, self.sequences)
                model = self.base_model(n_components, X_train, lengths_train)
                if model is not None:
                    X_test, lengths_test = combine_sequences(test_indices, self.sequences)
                    cv_scores.append(model.score(X_test, lengths_test))
            score = np.mean(cv_scores)
            if score > optimal_score:
                total_model = self.base_model(n_components)
                if total_model is not None:
                    optimal_score = score
                    optimal_model = total_model
        return optimal_model
