
import pickle
import numpy as np

from gandgai.regressor.regressor import Regressor
from gandgai.regressor.ridgeregforest import RidgeRegForest


class TestRidgeRegForest_1:
    def setUp(self):
        rng = np.random.RandomState(19)
        x = (rng.rand(5000, 5) - 0.5) * 4.0
        y = np.array([
            (x[:, 0] - x[:, 1])**2 + 2*x[:, 2],
            x[:, 1] + x[:, 3]**3 - 2*x[:, 4],
            np.min(x, axis=1),
        ]).T
        self.x = x
        self.y = y + 0.1 * rng.randn(*y.shape)

    def tearDown(self):
        self.x = None
        self.y = None

    def test_max_depth_with_bootstrap(self):
        x = self.x[:-1000, :]
        y = self.y[:-1000, :]
        rrf = RidgeRegForest(n_estimators=20, max_depth=4, random_state=13)
        rrf.fit(x, y)
        train_yhat = rrf.predict(x)
        train_mse = Regressor.mse(train_yhat, y)
        assert np.allclose(train_mse, 0.446658), train_mse
        valid_yhat = rrf.predict(self.x[-1000:, :])
        valid_mse = Regressor.mse(valid_yhat, self.y[-1000:, :])
        assert np.allclose(valid_mse, 0.464231), valid_mse
        
    def test_max_depth_with_bootstrap_max_samples(self):
        x = self.x[:-1000, :]
        y = self.y[:-1000, :]
        rrf = RidgeRegForest(n_estimators=20, max_depth=4, max_samples='n/2', random_state=13)
        rrf.fit(x, y)
        train_yhat = rrf.predict(x)
        train_mse = Regressor.mse(train_yhat, y)
        assert np.allclose(train_mse, 0.425346), train_mse
        valid_yhat = rrf.predict(self.x[-1000:, :])
        valid_mse = Regressor.mse(valid_yhat, self.y[-1000:, :])
        assert np.allclose(valid_mse, 0.441351), valid_mse
        
    def test_max_depth_without_bootstrap(self):
        x = self.x[:-1000, :]
        y = self.y[:-1000, :]
        rrf = RidgeRegForest(n_estimators=20, bootstrap=False, max_depth=4, random_state=13)
        rrf.fit(x, y)
        train_yhat = rrf.predict(x)
        train_mse = Regressor.mse(train_yhat, y)
        assert np.allclose(train_mse, 0.521047), train_mse
        valid_yhat = rrf.predict(self.x[-1000:, :])
        valid_mse = Regressor.mse(valid_yhat, self.y[-1000:, :])
        assert np.allclose(valid_mse, 0.536351), valid_mse

    def test_pickle(self):
        x = self.x[:-1000, :]
        y = self.y[:-1000, :]
        rrf = RidgeRegForest(n_estimators=10, max_leaf_nodes=10, random_state=13)
        rrf.fit(x, y)
        pickled_rrf = pickle.dumps(rrf)
        rrf = pickle.loads(pickled_rrf)
        train_yhat = rrf.predict(x)
        train_mse = Regressor.mse(train_yhat, y)
        assert np.allclose(train_mse, 0.349665), train_mse
        valid_yhat = rrf.predict(self.x[-1000:, :])
        valid_mse = Regressor.mse(valid_yhat, self.y[-1000:, :])
        assert np.allclose(valid_mse, 0.364856), valid_mse


class TestRidgeRegForest_2:
    def setUp(self):
        z = np.array(range(100000), dtype=float)/80000. - 0.25
        self.x = np.array([
            z,
            z*0.0 - 1.0,
            z**2,
            3*z**2-0.3,
            np.maximum(0.0, 4*z-0.2),
            np.minimum(0.0, 3*z-0.3),
            np.abs(z-0.1),
            np.log(1.0 + z**2),
        ]).T
        self.y = np.array([
            self.x[:, 0] + 2*self.x[:, 3],
            self.x[:, 1]**2 + 3*self.x[:, 0]*(self.x[:, 2]**3),
            np.max(self.x**3, axis=1),
            np.min(self.x, axis=1),
        ]).T

    def tearDown(self):
        self.x = None
        self.y = None

    def test_max_leaf_nodes(self):
        x = self.x
        y = self.y
        rrf = RidgeRegForest(n_estimators=5, max_leaf_nodes=100, random_state=17)
        rrf.fit(x, y)
        yhat = rrf.predict(x)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 1.807e-05), mse
