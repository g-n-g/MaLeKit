
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
        assert np.allclose(train_mse, 0.421036), train_mse
        valid_yhat = rrf.predict(self.x[-1000:, :])
        valid_mse = Regressor.mse(valid_yhat, self.y[-1000:, :])
        assert np.allclose(valid_mse, 0.434453), valid_mse
        
    def test_max_depth_with_bootstrap_max_samples(self):
        x = self.x[:-1000, :]
        y = self.y[:-1000, :]
        rrf = RidgeRegForest(n_estimators=20, max_depth=4, max_samples='n/2', random_state=13)
        rrf.fit(x, y)
        train_yhat = rrf.predict(x)
        train_mse = Regressor.mse(train_yhat, y)
        assert np.allclose(train_mse, 0.4001958), train_mse
        valid_yhat = rrf.predict(self.x[-1000:, :])
        valid_mse = Regressor.mse(valid_yhat, self.y[-1000:, :])
        assert np.allclose(valid_mse, 0.4165611), valid_mse
        
    def test_max_depth_without_bootstrap(self):
        x = self.x[:-1000, :]
        y = self.y[:-1000, :]
        rrf = RidgeRegForest(n_estimators=20, bootstrap=False, max_depth=4, random_state=13)
        rrf.fit(x, y)
        train_yhat = rrf.predict(x)
        train_mse = Regressor.mse(train_yhat, y)
        assert np.allclose(train_mse, 0.5271), train_mse
        valid_yhat = rrf.predict(self.x[-1000:, :])
        valid_mse = Regressor.mse(valid_yhat, self.y[-1000:, :])
        assert np.allclose(valid_mse, 0.535054), valid_mse

    def test_pickle(self):
        x = self.x[:-1000, :]
        y = self.y[:-1000, :]
        rrf = RidgeRegForest(n_estimators=10, max_leaf_nodes=10, random_state=13)
        rrf.fit(x, y)
        pickled_rrf = pickle.dumps(rrf)
        rrf = pickle.loads(pickled_rrf)
        train_yhat = rrf.predict(x)
        train_mse = Regressor.mse(train_yhat, y)
        assert np.allclose(train_mse, 0.313538), train_mse
        valid_yhat = rrf.predict(self.x[-1000:, :])
        valid_mse = Regressor.mse(valid_yhat, self.y[-1000:, :])
        assert np.allclose(valid_mse, 0.327077), valid_mse
