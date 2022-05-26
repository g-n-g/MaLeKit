
import numpy as np

from gandgai.regressor.regressor import Regressor
from gandgai.regressor.ridgeregtree import RidgeRegTree


class RidgeRegForest(Regressor):
    """Ridge regression forest.

    Attributes
    ----------
    n_estimators: positive int, or float, or str
        Number of trees to train in the forest.

        A float number is interpreted as the fraction of the number of samples
        ceiled to the closest integer. String format is evaluated with n being
        the number of samples and d being the number of features. Numpy
        functions can be used, for example 'd*np.log(n)', and the result is
        ceiled to the closest integer.

    bootstrap: boolean
        If False, each tree training uses all the training data. Otherwise,
        if True, max_samples number of samples are drawn randomly with
        replacement for each tree training (as for bagging).

    max_samples: None, or int, or float, or str
        Specifies the number of samples to be drawn randomly (with replacement)
        for each tree training. It only has effect if bootstrap is True.

        None means as many samples as in the original dataset. A float number
        is interpreted as the fraction of the number of samples in the original
        dataset ceiled to the closest integer. String format is evaluated with
        n being the number of samples and d being the number of features. Numpy
        functions can be used, for example 'd*np.log(n)', and the result is
        ceiled to the closest integer.

    The rest of the attributes are the same as for
    gandgai.regressor.RidgeRegTree.
    """
    def __init__(
        self,
        n_estimators=100,
        bootstrap=True,
        max_samples=None,
        # tree parameters
        alpha=1e-6,
        n_knots=10,
        jitter_tol=1e-6,
        max_features=None,
        n_random_direction=0,
        max_depth=None,
        max_leaf_nodes=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_mse_decrease=0.0,
        # other parameters:
        random_state=None,
        verbose=0,
        log_func=print,
        **other_params
    ):
        super().__init__(**other_params)
        #---
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        #---
        self.tree_params = {
            'alpha': alpha,
            'n_knots': n_knots,
            'jitter_tol': jitter_tol,
            'max_features': max_features,
            'n_random_direction': n_random_direction,
            'max_depth': max_depth,
            'max_leaf_nodes': max_leaf_nodes,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'min_mse_decrease': min_mse_decrease,
        }
        #---
        self.random_state = (
            random_state if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(random_state)
        )
        self.verbose = verbose
        self.log_func = log_func

    def _fit(self, x, y):
        n, d = x.shape

        n_estimators = Regressor.parse_int_param(self.n_estimators, n, d, 1)
        max_samples = Regressor.parse_int_param(self.max_samples, n, d, 1, n, n)

        self.estimators_ = []
        for estimator_idx in range(n_estimators):
            seed = self.random_state.randint(0, 1e8)
            if self.verbose >= 1:
                self.log_func('RRF, training estimator: {}/{}, seed: {}'.format(
                    1+estimator_idx, n_estimators, seed,
                ))
            if self.bootstrap:
                sample_idx = self.random_state.choice(n, max_samples, replace=True)
                sample_idx.sort()
                train_x = x[sample_idx, :]
                train_y = y[sample_idx, :]
            else:
                train_x = x
                train_y = y

            estimator = RidgeRegTree(
                random_state=seed,
                verbose=max(0, self.verbose-1),
                log_func=self.log_func,
                **self.tree_params
            )
            estimator.fit(train_x, train_y)
            self.estimators_.append(estimator)

    def _predict(self, x):
        yhat = np.zeros((x.shape[0], max(1, self.n_outputs_)))
        for estimator in self.estimators_:
            yhat += estimator.predict(x)
        yhat /= len(self.estimators_)
        return yhat
