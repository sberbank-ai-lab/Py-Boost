import cupy as cp

try:
    from cuml import TruncatedSVD, Handle
except ImportError:
    pass

from ..callbacks.callback import Callback


class GradProxy(Callback):

    def __call__(self, grad, hess):
        return grad, hess


class BestOutputProxy(GradProxy):

    def __init__(self, topk=1):
        self.topk = topk

    def __call__(self, grad, hess, sample_weight=None):
        best_idx = (grad ** 2).mean(axis=0).argsort()[-self.topk:]
        grad = grad[:, best_idx]

        return grad, hess


class SVDProxy(GradProxy):

    def __init__(self, sample=None, batch_size=10240, **svd_params):
        self.svd_params = {**{'algorithm': 'jacobi', 'n_components': 5, 'n_iter': 5}, **svd_params}
        self.n = 0
        self.sample = sample

    def __call__(self, grad, hess):
        handle = Handle()

        svd = TruncatedSVD(handle=handle, output_type='cupy', **self.svd_params)
        sub_grad = grad
        if (self.sample is not None) and (grad.shape[0] > self.sample):
            idx = cp.arange(grad.shape[0], dtype=cp.int32)
            cp.random.shuffle(idx)
            sub_grad = grad[idx[:self.sample]]

        svd.fit(sub_grad)
        proxy_grad = svd.transform(grad)

        return proxy_grad, hess

    def after_iteration(self, build_info):
        build_info['mempool'].free_all_blocks()


class BestRandomProxy(GradProxy):

    def __init__(self, n=10, smooth=0.1):
        self.n = n
        self.smooth = smooth

    def __call__(self, grad, hess):
        best_idx = (grad ** 2).mean(axis=0)
        pi = best_idx / best_idx.sum()
        pi = self.smooth * cp.ones_like(pi) / grad.shape[1] + (1 - self.smooth) * pi

        gg = grad / cp.sqrt(self.n * pi)
        rand_idx = cp.random.choice(cp.arange(grad.shape[1]), size=self.n, replace=True, p=pi)
        grad = gg[:, rand_idx]

        return grad, hess
