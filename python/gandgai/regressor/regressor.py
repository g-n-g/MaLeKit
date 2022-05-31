
import numpy as np
from scipy import linalg
from collections import namedtuple


LinearModelParams = namedtuple("LinearModelParams", "slope bias")


class Regressor:
    """Abstract class of regressors.
    """
    def __init__(self):
        pass

    def _fit(self, x, y):
        raise NotImplementedError('Method _fit() is called in abstract Regressor class!')

    def fit(self, x, y):
        assert x.shape[0] == y.shape[0]
        self.n_features_in_ = x.shape[1]
        if len(y.shape) == 1:
            y = y[:, np.newaxis]
            self.n_outputs_ = 0
        else:
            self.n_outputs_ = y.shape[1]
        return self._fit(x, y)

    def _predict(self, x):
        raise NotImplementedError('Method _predict() is called in abstract Regressor class!')

    def predict(self, x):
        assert x.shape[1] == self.n_features_in_
        return self._predict(x)

    def score(self, x, y):
        """Computes the coefficient of determination (R squared):
        https://en.wikipedia.org/wiki/Coefficient_of_determination
        """
        if len(y.shape) < 2:
            y = y[:, np.newaxis]
            assert y.shape[1] == max(1, self.n_outputs_)
        yhat = self.predict(x)
        if len(yhat.shape) < 2:
            yhat = yhat[:, np.newaxis]
        yhat -= y
        np.square(yhat, out=yhat)
        ssq = np.sum(yhat)
        yhat[:] = y
        yhat -= np.mean(y, axis=0)[np.newaxis, :]
        np.square(yhat, out=yhat)
        return 1.0 - ssq/np.sum(yhat)

    @staticmethod
    def parse_int_param(
        spec, n, d,
        min_value=None, max_value=None, default=None, allow_none=False, float_ref=None,
    ):
        """Returns the integer parameter value based on its specification.

        Examples
        --------
        >>> Regressor.parse_int_param(7, 100, 3)
        7
        >>> Regressor.parse_int_param(0.046, 100, 3, min_value=3)
        5
        >>> Regressor.parse_int_param('d*np.log(n)', 100, 3)
        14
        """
        if spec is None:
            value = default
        elif isinstance(spec, int):
            value = spec
        elif isinstance(spec, float):
            if float_ref is None:
                float_ref = n
            value = int(np.ceil(spec * float_ref))
        elif isinstance(spec, str):
            value = int(np.ceil(eval(spec)))
        else:
            raise AttributeError('Invalid int param value: {}'.format(spec))
        if value is None:
            assert allow_none, 'None int param value is not allowed!'
        else:
            if min_value is not None:
                assert value >= min_value, 'Too low value: {} < {}'.format(value, min_value)
            if max_value is not None:
                assert value <= max_value, 'Too high value: {} > {}'.format(value, max_value)
        return value

    @staticmethod
    def parse_float_param(
        spec, n, d,
        min_value=None, max_value=None, default=None, allow_none=False,
    ):
        """Returns the float parameter value based on its specification.

        Examples
        --------
        >>> Regressor.parse_float_param(7.1, 100, 5)
        7.1
        >>> Regressor.parse_float_param(4.6, 100, 5, min_value=3)
        4.6
        >>> np.round(Regressor.parse_float_param('d*np.log(n)', 100, 3), decimals=4)
        13.8155
        """
        if spec is None:
            value = default
        elif isinstance(spec, float):
            value = spec
        elif isinstance(spec, str):
            value = eval(spec)
        else:
            raise AttributeError('Invalid float param value: {}'.format(spec))
        if value is None:
            assert allow_none, 'None float param value is not allowed!'
        else:
            if min_value is not None:
                assert value >= min_value, 'Too low value: {} < {}'.format(value, min_value)
            if max_value is not None:
                assert value <= max_value, 'Too high value: {} > {}'.format(value, max_value)
        return value

    @staticmethod
    def ridge_fit(
        x, y, alpha,
        xx=None, xy=None, xsum=None, ysum=None, tmp_d=None, tmp_dd=None, tmp_dk=None,
    ):
        n = float(x.shape[0])
        one_per_sqrtn = 1.0 / np.sqrt(n)
        if xx is None:
            xbar = np.sum(x, axis=0)
            ybar = np.sum(y, axis=0)
            xbar *= one_per_sqrtn
            ybar *= one_per_sqrtn
            xx = x.T.dot(x)
            xy = x.T.dot(y)
            xx -= np.outer(xbar, xbar)
            xy -= np.outer(xbar, ybar)
        else:
            xbar = np.multiply(xsum, one_per_sqrtn, out=tmp_d)
            ybar = ysum * one_per_sqrtn  # we need the copy here
            xx = xx - np.outer(xbar, xbar, out=tmp_dd)
            xy = xy - np.outer(xbar, ybar, out=tmp_dk)
        xx.flat[::xx.shape[1]+1] += alpha * n
        slope = linalg.solve(
            xx, xy,
            assume_a='pos', check_finite=False,
            overwrite_a=True, overwrite_b=True,
        )
        ybar -= xbar.dot(slope)
        ybar *= one_per_sqrtn
        return LinearModelParams(slope, ybar)

    @staticmethod
    def mse(yhat, y):
        """Calculates the mean squared error between two matrices:
        https://en.wikipedia.org/wiki/Mean_squared_error
        """
        diff = yhat - y
        np.square(diff, out=diff)
        return np.mean(diff)

    @staticmethod
    def linear_predict(x, linear_model_params):
        yhat = x.dot(linear_model_params.slope)
        yhat += linear_model_params.bias
        return yhat 
