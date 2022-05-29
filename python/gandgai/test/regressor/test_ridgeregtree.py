
import pickle
import numpy as np

from gandgai.regressor.regressor import LinearModelParams, Regressor
from gandgai.regressor.ridgeregtree import RidgeRegTree, TreeNode


class TestRidgeRegTree_1:
    def setUp(self):
        self.msg = ""
        self.x = np.array([
            [1., -0.2, 1., 1.], [2., -0.4, 2., 2.],
            [3., -0.6, 3., 3.], [4., -0.8, 5., 5.],
            [5., -1.0, 7., 8.], [6., -1.2, 11., 13.],
        ])
        self.y = np.array([
            [4., -2., 1.], [5., -3., 0.], [6., -5., 1.],
            [7., -7., 4.], [8., -9., 9.], [9., -11., 16.],
        ])

    def tearDown(self):
        self.x = None
        self.y = None

    def log(self, s):
        self.msg += '\n' + s

    def test_dim0(self):
        x = self.x
        y = self.y[:, 0]
        rrt = RidgeRegTree()
        rrt.fit(x, y)
        assert len(rrt.tree_) == 1, rrt.tree_
        assert isinstance(rrt.tree_[0], LinearModelParams), rrt.tree_[0]
        yhat = rrt.predict(x)
        assert np.allclose(y, yhat)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 0.0), mse
        score = rrt.score(x, y)
        assert np.allclose(score, 1.0), score

    def test_dim1(self):
        x = self.x
        y = self.y[:, 1]
        rrt = RidgeRegTree(verbose=1, log_func=self.log)
        rrt.fit(x, y)
        assert '\nRRT, depth: 1, n_leaf_nodes: 1\nRRT, depth: 2, n_leaf_nodes: 2' == self.msg, \
            self.msg
        assert len(rrt.tree_) == 3, rrt.tree_
        assert isinstance(rrt.tree_[0], TreeNode), rrt.tree_[0]
        assert isinstance(rrt.tree_[1], LinearModelParams), rrt.tree_[1]
        assert isinstance(rrt.tree_[2], LinearModelParams), rrt.tree_[2]
        yhat = rrt.predict(x)
        assert np.allclose(y, yhat)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 0.0), mse
        score = rrt.score(x, y)
        assert np.allclose(score, 1.0), score

    def test_ally(self):
        x = self.x
        y = self.y
        rrt = RidgeRegTree()
        rrt.fit(x, y)
        assert len(rrt.tree_) == 11, rrt.tree_
        yhat = rrt.predict(x)
        assert np.allclose(y, yhat)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 0.0), mse
        score = rrt.score(x, y)
        assert np.allclose(score, 1.0), score


class TestRidgeRegTree_2:
    def setUp(self):
        z = np.array(range(2000), dtype=float)/1000. - 0.25
        self.x = np.array([
            z,
            z*0.0 - 1.0,
            3*z**2-0.3,
            np.maximum(0.0, 4*z-0.2),
            np.abs(z-0.1),
        ]).T
        self.y = np.array([
            self.x[:, 0] + 2*self.x[:, 3],
            self.x[:, 1]**2 + 3*self.x[:, 0]*(self.x[:, 2]**3),
            np.max(self.x**3, axis=1),
        ]).T

    def tearDown(self):
        self.x = None
        self.y = None

    def test_max_depth(self):
        x = self.x
        y = self.y
        rrt = RidgeRegTree(max_depth=4, random_state=13)
        rrt.fit(x, y)
        assert len(rrt.tree_) == 15, len(rrt.tree_)
        yhat = rrt.predict(x)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 0.886653), mse

    def test_max_leaf_nodes(self):
        x = self.x.copy()
        y = self.y.copy()
        rrt = RidgeRegTree(max_leaf_nodes=7, random_state=17)
        rrt.fit(x, y)
        assert len(rrt.tree_) == 13, len(rrt.tree_)
        yhat = rrt.predict(x)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 1.341134), mse
        assert np.allclose(x, self.x)
        assert np.allclose(y, self.y)

    def test_min_samples_leaf(self):
        x = self.x
        y = self.y
        rrt = RidgeRegTree(min_samples_leaf=100, random_state=19, verbose=2)
        rrt.fit(x, y)
        assert len(rrt.tree_) == 29, len(rrt.tree_)
        yhat = rrt.predict(x)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 1.525537), mse

    def test_max_features(self):
        x = self.x
        y = self.y
        rrt = RidgeRegTree(min_samples_leaf=100, random_state=19, max_features=2)
        rrt.fit(x, y)
        assert len(rrt.tree_) == 27, len(rrt.tree_)
        yhat = rrt.predict(x)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 1.286623), mse

    def test_random_directions(self):
        x = self.x.copy()
        y = self.y.copy()
        rrt = RidgeRegTree(min_samples_leaf=100, random_state=19,
                           max_features=0, n_random_direction=2)
        rrt.fit(x, y)
        assert len(rrt.tree_) == 25, len(rrt.tree_)
        yhat = rrt.predict(x)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 1.532984), mse
        assert np.allclose(x, self.x)
        assert np.allclose(y, self.y)

    def test_n_knots(self):
        x = self.x
        y = self.y
        rrt = RidgeRegTree(min_samples_leaf=100, random_state=19, n_knots='5*d-2')
        rrt.fit(x, y)
        assert len(rrt.tree_) == 27, len(rrt.tree_)
        yhat = rrt.predict(x)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 1.5261516), mse

    def test_min_samples_split(self):
        x = self.x
        y = self.y
        rrt = RidgeRegTree(min_samples_split=300, random_state=23)
        rrt.fit(x, y)
        assert len(rrt.tree_) == 19, len(rrt.tree_)
        yhat = rrt.predict(x)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 2.091384), mse

    def test_min_mse_decrease(self):
        x = self.x
        y = self.y
        rrt = RidgeRegTree(min_mse_decrease=0.25, random_state=29)
        rrt.fit(x, y)
        assert len(rrt.tree_) == 15, len(rrt.tree_)
        yhat = rrt.predict(x)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 0.886653), mse

    def test_pickle(self):
        x = self.x
        y = self.y
        rrt = RidgeRegTree(max_depth=5, max_leaf_nodes=10, random_state=31)
        rrt.fit(x, y)
        assert len(rrt.tree_) == 19, len(rrt.tree_)
        pickled_rrt = pickle.dumps(rrt)
        rrt = pickle.loads(pickled_rrt)
        yhat = rrt.predict(x)
        mse = Regressor.mse(yhat, y)
        assert np.allclose(mse, 0.781311), mse
