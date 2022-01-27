"""Defines sketching strategies to simplify multioutput scoring function calculation"""

import cupy as cp

try:
    from cuml import TruncatedSVD
except ImportError:
    pass

from ..callbacks.callback import Callback


class GradSketch(Callback):
    """Basic class for sketching strategy.
    It should implement __call__ method.
    """

    def __call__(self, grad, hess):
        """Method receive raw grad and hess matrices and output new ones that will be used in the tree structure search

        Args:
            grad: cp.ndarray, gradients
            hess: cp.ndarray, hessians

        Returns:
            cp.ndarray, sketched grad
            cp.ndarray, sketched hess
        """
        return grad, hess


class TopOutputsSketch(GradSketch):
    """TopOutputs sketching. Use only gradient columns with the highest L2 norm"""

    def __init__(self, topk=1):
        """

        Args:
            topk: int, top outputs to use
        """
        self.topk = topk

    def __call__(self, grad, hess):
        best_idx = (grad ** 2).mean(axis=0).argsort()[-self.topk:]
        grad = grad[:, best_idx]

        if hess.shape[1] > 1:
            hess = hess[:, best_idx]

        return grad, hess


class SVDSketch(GradSketch):
    """SVD Sketching. Truncated SVD is used to reduce grad dimensions."""

    def __init__(self, sample=None, **svd_params):
        """

        Args:
            sample: int, subsample to speed up SVD fitting
            **svd_params: dict, SVD params, see cuml.TruncatedSVD docs
        """
        self.svd_params = {**{'algorithm': 'jacobi', 'n_components': 5, 'n_iter': 5}, **svd_params}
        self.sample = sample
        self.svd = None

    def before_train(self, build_info):
        self.svd = TruncatedSVD(output_type='cupy', **self.svd_params)

    def __call__(self, grad, hess):

        sub_grad = grad
        if (self.sample is not None) and (grad.shape[0] > self.sample):
            idx = cp.arange(grad.shape[0], dtype=cp.int32)
            cp.random.shuffle(idx)
            sub_grad = grad[idx[:self.sample]]

        self.svd.fit(sub_grad)
        grad = self.svd.transform(grad)

        if hess.shape[1] > 1:
            hess = self.svd.transform(hess)
            hess = cp.clip(hess, 0.01)

        return grad, hess

    def after_iteration(self, build_info):
        """Free memory to avoid OOM.

        Args:
            build_info: dict

        Returns:

        """
        build_info['mempool'].free_all_blocks()

    def after_train(self, build_info):
        self.svd = None


class RandomSamplingSketch(GradSketch):
    """RandomSampling Sketching. Gradient columns are randomly sampled with probabilities."""

    def __init__(self, n=10, smooth=0.1, replace=True):
        """

        Args:
            n: int, n outputs to select
            smooth: float, 0 stands for probabilities proportionally to the sum of squares, 1 stands for uniform.
                (0, 1) stands for tradeoff
        """
        self.n = n
        self.smooth = smooth
        self.replace = replace

    def __call__(self, grad, hess):
        best_idx = (grad ** 2).mean(axis=0)
        pi = best_idx / best_idx.sum()
        pi = self.smooth * cp.ones_like(pi) / grad.shape[1] + (1 - self.smooth) * pi

        gg = grad / cp.sqrt(self.n * pi)
        rand_idx = cp.random.choice(cp.arange(grad.shape[1]), size=self.n, replace=self.replace, p=pi)
        grad = gg[:, rand_idx]

        if hess.shape[1] > 1:
            hess = hess[:, rand_idx]

        return grad, hess


class RandomProjectionSketch(GradSketch):
    """Random projection sketch"""

    def __init__(self, n, norm=True):
        """

        Args:
            n: int, number of output dimensions
            norm: if True use normal distribution, otherwise +1/-1
        """
        self.k = n
        self.norm = norm

    def __call__(self, grad, hess):

        if self.norm:
            P = cp.random.randn(grad.shape[1], self.k, dtype=cp.float32)
        else:
            P = (cp.random.rand(grad.shape[1], self.k, dtype=cp.float32) > .5).astype(cp.float32) * 2 - 1

        P /= cp.sqrt(1 / self.k)

        grad = cp.dot(grad, P)

        if hess.shape[1] > 1:
            hess = cp.dot(hess, P)
            hess = cp.clip(hess, 0.01)

        return grad, hess
