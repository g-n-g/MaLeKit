
import heapq
import numpy as np
from collections import namedtuple

from gandgai.regressor.regressor import Regressor, LinearModelParams


TreeNode = namedtuple("TreeNode", "data child_idx")
CoordSplit = namedtuple("CoordSplit", "coord cut_value")
VectorSplit = namedtuple("VectorSplit", "vector cut_value")


class RidgeRegTree(Regressor):
    """Ridge regression tree.

    Attributes
    ----------
    alpha : nonnegative float, or str
        Ridge regularization parameter.

    n_knots: positive int, or float in (0,1], or str
        Number of knots to check along a split direction.

        A float number is interpreted as the fraction of the number of samples
        ceiled to the closest integer. String format is evaluated with n being
        the number of samples and d being the number of features. Numpy
        functions can be used, for example 'd*np.log(n)', and the result is
        ceiled to the closest integer.

    jitter_tol: positive float
        Amount of noise to be introduced when splitting along almost constant
        directions.

    max_features: None, or nonegative int, or float in [0,1], or str
        Maximum number of features to be considered for each split.

        The order of features are permuted in each iteration even if
        max_features specifies all features, so ties might be broken
        differently in different iterations.

        None means all available features. A float number is interpreted as the
        fraction of the number of samples ceiled to the closest integer. String
        format is evaluated with n being the number of samples and d being the
        number of features. Numpy functions can be used, for example
        'd*np.log(n)', and the result is ceiled to the closest integer.

        It can be set to zero in which case feature based (coordinate-wise)
        splitting will not be performed (in this case n_random_direction
        has to be set positive).

    n_random_direction: nonnegative int, or float in [0,1], or str
        Number of random directions to be searched along for each split.

        A float number is interpreted as the fraction of the number of samples
        ceiled to the closest integer. String format is evaluated with n being
        the number of samples and d being the number of features. Numpy
        functions can be used, for example 'd*np.log(n)', and the result is
        ceiled to the closest integer.

    max_depth: None, or nonegative int, or float in [0,1], or str
        Maximum tree depth.

        None means there is no maximum. A float number is interpreted as the
        fraction of the number of samples ceiled to the closest integer. String
        format is evaluated with n being the number of samples and d being the
        number of features. Numpy functions can be used, for example
        'd*np.log(n)', and the result is ceiled to the closest integer.

    max_leaf_nodes: None, or nonegative int, or float in [0,1], or str
        Maximum number of leaf nodes.

        None means there is no maximum. A float number is interpreted as the
        fraction of the number of samples ceiled to the closest integer. String
        format is evaluated with n being the number of samples and d being the
        number of features. Numpy functions can be used, for example
        'd*np.log(n)', and the result is ceiled to the closest integer.

    min_samples_split: positive int, or float in [0,1], or str
        Minimum number of samples at a node to be considered for a split.

        A float number is interpreted as the fraction of the number of samples
        ceiled to the closest integer. String format is evaluated with n being
        the number of samples and d being the number of features. Numpy
        functions can be used, for example 'd*np.log(n)', and the result is
        ceiled to the closest integer.

    min_samples_leaf: positive int, or float in [0,1], or str
        Minimum number of samples at leaf nodes.

        A float number is interpreted as the fraction of the number of samples
        ceiled to the closest integer. String format is evaluated with n being
        the number of samples and d being the number of features. Numpy
        functions can be used, for example 'd*np.log(n)', and the result is
        ceiled to the closest integer.

    min_mse_decrease: float, or str
        Minimum MSE decrease to be achieved by a new split.

        String format is evaluated with n being the number of samples and d
        being the number of features. Numpy functions can be used, for example
        'd*np.log(n)', and the result is ceiled to the closest integer.

    random_state: int, or np.random.RandomState
        Initial state of the random number generator used by the algorithm.

    verbose: int
        Verbosity level.

    log_func: logger function of a string argument
        Logging function for verbose operation.

    other_params: inherited from gandgai.regressor.Regressor
    """
    def __init__(
        self,
        alpha=1e-6,
        # splitting parameters:
        n_knots=9,
        jitter_tol=1e-6,
        max_features=None,
        n_random_direction=0,
        # tree building parameters:
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
        self.alpha = alpha
        self.n_knots = n_knots
        self.jitter_tol = jitter_tol
        self.max_features = max_features
        self.n_random_direction = n_random_direction
        #---
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_mse_decrease = min_mse_decrease
        #---
        self.random_state = (
            random_state if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(random_state)
        )
        self.verbose = verbose
        self.log_func = log_func

    @staticmethod
    def fit_and_eval(x, y, alpha, xx=None, xy=None, xsum=None, ysum=None):
        lmp = Regressor.ridge_fit(x, y, alpha, xx, xy, xsum, ysum)
        z = Regressor.linear_predict(x, lmp)
        z -= y
        np.square(z, out=z)
        return (lmp, np.sum(z))

    CandidateData = namedtuple(
        "CandidateData", "dssq idx split left_idx, left right_idx right",
    )

    def get_scores(self, scores):
        min_score = np.min(scores)
        score_range = np.max(scores) - min_score
        if score_range < self.jitter_tol:
            noise = self.random_state.randn(*scores.shape)
            noise *= self.jitter_tol
            scores = scores + noise
            min_score = np.min(scores)
            score_range = np.max(scores) - min_score
        return scores, min_score, score_range

    def eval_split(
        self, raw_scores, make_split_func,
        sample_idx, min_samples_leaf, knots,
        x, y, alpha, node_idx, node_ssq, best_cand_data,
    ):
        scores, min_score, score_range = self.get_scores(raw_scores)
        cut_bin = np.digitize((scores - min_score) / score_range, knots, right=True)
        cut_xx = []
        cut_xy = []
        cut_xsum = []
        cut_ysum = []
        masks = []
        left_xx = None
        left_xy = None
        left_xsum = None
        left_ysum = None
        right_xx = None
        right_xy = None
        right_xsum = None
        right_ysum = None
        for cut_idx in range(len(knots)+1):
            mask = cut_bin == cut_idx
            idx = sample_idx[mask]
            xidx = x[idx, :]
            yidx = y[idx, :]
            cut_xx.append(xidx.T.dot(xidx))
            cut_xy.append(xidx.T.dot(yidx))
            cut_xsum.append(np.sum(xidx, axis=0))
            cut_ysum.append(np.sum(yidx, axis=0))
            if cut_idx == 0:
                left_xx = cut_xx[0].copy()
                left_xy = cut_xy[0].copy()
                left_xsum = cut_xsum[0].copy()
                left_ysum = cut_ysum[0].copy()
            elif cut_idx == 1:
                right_xx = cut_xx[-1].copy()
                right_xy = cut_xy[-1].copy()
                right_xsum = cut_xsum[-1].copy()
                right_ysum = cut_ysum[-1].copy()
                mask |= masks[-1]
            else:
                right_xx += cut_xx[-1]
                right_xy += cut_xy[-1]
                right_xsum += cut_xsum[-1]
                right_ysum += cut_ysum[-1]
                mask |= masks[-1]
            masks.append(mask)
        for cut_idx in range(len(knots)):
            if cut_idx > 0:
                left_xx += cut_xx[cut_idx]
                left_xy += cut_xy[cut_idx]
                left_xsum += cut_xsum[cut_idx]
                left_ysum += cut_ysum[cut_idx]
                right_xx -= cut_xx[cut_idx]
                right_xy -= cut_xy[cut_idx]
                right_xsum -= cut_xsum[cut_idx]
                right_ysum -= cut_ysum[cut_idx]
            mask = masks[cut_idx]
            left_idx = sample_idx[mask]
            if len(left_idx) < min_samples_leaf:
                continue
            right_idx = sample_idx[~mask]
            if len(right_idx) < min_samples_leaf:
                continue

            cand_left = RidgeRegTree.fit_and_eval(
                x[left_idx, :], y[left_idx, :], alpha,
                left_xx, left_xy, left_xsum, left_ysum,
            )
            cand_right = RidgeRegTree.fit_and_eval(
                x[right_idx, :], y[right_idx, :], alpha,
                right_xx, right_xy, right_xsum, right_ysum,
            )
            cand_dssq = cand_left[1] + cand_right[1] - node_ssq
            if cand_dssq < best_cand_data.dssq:
                split = make_split_func(min_score + knots[cut_idx] * score_range)
                best_cand_data = RidgeRegTree.CandidateData(
                    cand_dssq, node_idx, split, left_idx, cand_left, right_idx, cand_right,
                )
        return best_cand_data

    def _fit(self, x, y):
        n, d = x.shape

        alpha = Regressor.parse_float_param(self.alpha, n, d, 0)
        n_knots = Regressor.parse_int_param(self.n_knots, n, d, 1)
        max_depth = Regressor.parse_int_param(self.max_depth, n, d, 1, allow_none=True)
        max_features = Regressor.parse_int_param(self.max_features, n, d, 0, d, d)
        max_leaf_nodes = Regressor.parse_int_param(self.max_leaf_nodes, n, d, 0, allow_none=True)
        n_random_direction = Regressor.parse_int_param(self.n_random_direction, n, d, 0, None, 0)
        min_samples_split = Regressor.parse_int_param(self.min_samples_split, n, d, 0)
        min_samples_leaf = Regressor.parse_int_param(self.min_samples_leaf, n, d, 0)
        min_mse_decrease = Regressor.parse_float_param(self.min_mse_decrease, n, d, 0)
        if min_mse_decrease != 0.0:
            min_mse_decrease *= n

        cand_data_init = RidgeRegTree.CandidateData(
            -min_mse_decrease, None, None, None, None, None, None,
        )
        tree = [TreeNode(RidgeRegTree.fit_and_eval(x, y, alpha), None)]
        candidates = []
        depths = [1]
        sample_idxs = [np.arange(n)]
        coords = np.arange(d)
        n_leaf_nodes = 1
        knots = (np.arange(n_knots+2)/float(n_knots+1))[1:-1]
        while True:
            if self.verbose >= 1:
                self.log_func('RRT, depth: {}, n_leaf_nodes: {}'.format(
                    np.max(depths), n_leaf_nodes,
                ))
            if max_leaf_nodes is not None and n_leaf_nodes >= max_leaf_nodes:
                break
            if max_features > 0:
                self.random_state.shuffle(coords)

            tree_size = len(tree)
            for node_idx in range(max(0, tree_size-2), tree_size):
                sample_idx = sample_idxs[node_idx]
                if sample_idx is None or len(sample_idx) < min_samples_split:
                    continue
                if max_depth is not None and depths[node_idx] >= max_depth:
                    continue
                node_cand_data = cand_data_init
                node_ssq = tree[node_idx].data[1]
                xnode = x[sample_idx, :]
                ynode = y[sample_idx, :]
                if max_features > 0:
                    for coord in coords[:max_features]:
                        node_cand_data = self.eval_split(
                            xnode[:, coord],
                            lambda cut_value: CoordSplit(coord, cut_value),
                            sample_idx, min_samples_leaf, knots,
                            x, y, alpha, node_idx, node_ssq, node_cand_data,
                        )
                if n_random_direction > 0:
                    rand_dirs = self.random_state.randn(n, d)
                    rand_dirs /= np.linalg.norm(rand_dirs, axis=1)[:, np.newaxis]
                    for rand_dir_idx in range(n_random_direction):
                        rand_dir = rand_dirs[rand_dir_idx, :]
                        node_cand_data = self.eval_split(
                            xnode.dot(rand_dir),
                            lambda cut_value: VectorSplit(rand_dir, cut_value),
                            sample_idx, min_samples_leaf, knots,
                            x, y, alpha, node_idx, node_ssq, node_cand_data,
                        )
                if node_cand_data.idx is not None:
                    heapq.heappush(candidates, node_cand_data)
                    if self.verbose >= 2:
                        self.log_func(
                            'RRT, new candidate, node_idx: {}, dssq: {}'.format(
                                node_idx, node_cand_data.dssq,
                            ))

            if len(candidates) == 0:
                break  # stop because we did not find any valid candidate

            best_cand_data = heapq.heappop(candidates)
            tree[best_cand_data.idx] = TreeNode(best_cand_data.split, tree_size)
            sample_idxs[best_cand_data.idx] = None
            n_leaf_nodes += 1  # splitting replaces a leaf node by two
            # add left child
            tree.append(TreeNode(best_cand_data.left, None))
            depths.append(depths[best_cand_data.idx] + 1)
            sample_idxs.append(best_cand_data.left_idx)
            # add right child
            tree.append(TreeNode(best_cand_data.right, None))
            depths.append(depths[best_cand_data.idx] + 1)
            sample_idxs.append(best_cand_data.right_idx)

        tree = [node.data[0] if node.child_idx is None else node for node in tree]
        self.tree_ = tuple(tree)

    def _predict(self, x):
        tree = self.tree_
        yhat = np.empty((x.shape[0], max(1, self.n_outputs_)))
        yhat[:] = np.nan
        evals = [(0, np.arange(x.shape[0]))]
        while len(evals) > 0:
            node_idx, sample_idx = evals.pop()
            node = tree[node_idx]
            if isinstance(node, LinearModelParams):
                yhat[sample_idx] = Regressor.linear_predict(x[sample_idx, :], node)
            else:  # isinstance(node, TreeNode) is True
                node_data = node.data
                if isinstance(node_data, CoordSplit):
                    coord, cut_value = node_data
                    scores = x[sample_idx, coord]
                elif isinstance(node_data, VectorSplit):
                    rand_dir, cut_value = node_data
                    scores = x[sample_idx, :].dot(rand_dir)
                else:
                    raise RuntimeError('Invalid node: {}'.format(node))
                mask = (scores <= cut_value)
                idx = sample_idx[mask]
                if len(idx) > 0:
                    evals.append((node.child_idx, idx))
                idx = sample_idx[~mask]
                if len(idx) > 0:
                    evals.append((node.child_idx+1, idx))
        if self.n_outputs_ == 0:
            yhat = yhat[:, 0]
        return yhat
