"""Common losses and metrics"""

import cupy as cp
import numpy as np

from ..callbacks.callback import Callback


class Loss:

    def get_grad_hess(self, y_true, y_pred):
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        grad, hess = self.get_grad_hess(y_true, y_pred)
        return grad, hess

    def preprocess_input(self, y_true):
        return y_true

    def postprocess_output(self, y_pred):
        return y_pred

    def base_score(self, y_true):
        raise NotImplementedError

    def get_default_metric(self):
        raise NotImplementedError


class Metric:
    alias = 'score'

    def error(self, y_true, y_pred):
        raise ValueError('Pointwise error is not implemented for this metric')

    def __call__(self, y_true, y_pred, sample_weight=None):
        err = self.error(y_true, y_pred)
        shape = err.shape
        assert shape[0] == y_true.shape[0], 'Error shape should match target shape at first dim'

        if len(shape) == 1:
            err = err[:, cp.newaxis]

        if sample_weight is None:
            return err.mean()

        err = (err.mean(axis=1, keepdims=True) * sample_weight).sum() / sample_weight.sum()
        return err

    def compare(self, v0, v1):
        raise NotImplementedError

    def argmax(self, arr):

        best = arr[0]
        best_n = 0

        for n, val in enumerate(arr[1:], 1):
            if self.compare(val, best):
                best = val
                best_n = n

        return best_n


# regression losses

class MSELoss(Loss):

    def get_grad_hess(self, y_true, y_pred):
        return (y_pred - y_true), cp.ones((y_true.shape[0], 1), dtype=cp.float32)

    def base_score(self, y_true):
        return y_true.mean(axis=0)

    def get_default_metric(self):
        return RMSEMetric()


class RMSEMetric(Metric):
    alias = 'rmse'

    def error(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight) ** .5

    def compare(self, v0, v1):
        return v0 < v1


class R2Score(RMSEMetric):
    alias = 'R2_score'

    def __call__(self, y_true, y_pred, sample_weight=None):

        if sample_weight is not None:
            err = ((y_true - y_pred) ** 2 * sample_weight).sum(axis=0) / sample_weight.sum()
            std = ((y_true - y_true.mean(axis=0)) ** 2 * sample_weight).sum(axis=0) / sample_weight.sum()
        else:
            err = ((y_true - y_pred) ** 2).mean(axis=0)
            std = y_true.var(axis=0)

        return (1 - err / std).mean()

    def compare(self, v0, v1):
        return v0 > v1


# binary/multilabel losses

class BCELoss(Loss):

    def __init__(self, clip_value=1e-7):
        self.clip_value = clip_value

    def base_score(self, y_true):
        means = cp.clip(y_true.mean(axis=0), self.clip_value, 1 - self.clip_value)
        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        pred = 1 / (1 + cp.exp(-y_pred))
        pred = cp.clip(pred, self.clip_value, 1 - self.clip_value)
        grad = pred - y_true
        hess = pred * (1 - pred)

        return grad, hess

    def postprocess_output(self, y_pred):
        xp = np if type(y_pred) is np.ndarray else cp
        pred = 1 / (1 + xp.exp(-y_pred))
        pred = xp.clip(pred, self.clip_value, 1 - self.clip_value)

        return pred

    def get_default_metric(self):
        return BCEMetric()


class BCEMetric(Metric):
    alias = 'BCE'

    def error(self, y_true, y_pred):
        return -cp.log(y_true * y_pred + (1 - y_pred) * (1 - y_true))

    def compare(self, v0, v1):
        return v0 < v1


def softmax(x, clip_val=1e-5, xp=cp):
    exp_p = xp.exp(x - x.max(axis=1, keepdims=True))

    return xp.clip(exp_p / exp_p.sum(axis=1, keepdims=True), clip_val, 1 - clip_val)


# multiclass losses

ce_grad_kernel = cp.ElementwiseKernel(
    'T pred, raw S label, raw S nlabels, T factor',
    'T grad, T hess',

    """
    int y_pr = i % nlabels;
    int y_tr = label[i / nlabels];

    grad = pred - (float) (y_pr == y_tr);
    hess = pred * (1. - pred) * factor;

    """,
    "ce_grad_kernel"
)


def ce_grad(y_true, y_pred):
    factor = y_pred.shape[1] / (y_pred.shape[1] - 1)
    grad, hess = ce_grad_kernel(y_pred, y_true, y_pred.shape[1], factor)

    return grad, hess


class CrossEntropyLoss(Loss):

    def __init__(self, clip_value=1e-6):
        self.clip_value = clip_value

    def base_score(self, y_true):
        num_classes = int(y_true.max() + 1)
        hist = cp.zeros((num_classes,), dtype=cp.float32)

        return hist

    def get_grad_hess(self, y_true, y_pred):
        pred = softmax(y_pred, self.clip_value)
        grad, hess = ce_grad(y_true, pred)
        return grad, hess

    def postprocess_output(self, y_pred):
        xp = np if type(y_pred) is np.ndarray else cp
        return softmax(y_pred, self.clip_value, xp)

    def preprocess_input(self, y_true):
        return y_true[:, 0].astype(cp.int32)

    def get_default_metric(self):
        return CrossEntropyMetric()


class CrossEntropyMetric(Metric):
    alias = 'Crossentropy'

    def error(self, y_true, y_pred):
        return -cp.log(cp.take_along_axis(y_pred, y_true[:, cp.newaxis], axis=1))

    def compare(self, v0, v1):
        return v0 < v1


# multiclass metric for class label prediction

class MultiAccuracyMetric(Metric):
    alias = 'Accuracy'

    def error(self, y_true, y_pred):
        cl_pred = y_pred.argmax(axis=1)
        return (cl_pred == y_true).astype(cp.float32)

    def compare(self, v0, v1):
        return v0 > v1


class MultiMetric(Metric):

    def __init__(self, average='macro'):
        self.average = average

    @staticmethod
    def get_stats(y_true, y_pred, sample_weight=None, mode='f1'):

        if sample_weight is None:
            sample_weight = cp.ones(y_true.shape, dtype=cp.float32)
        else:
            sample_weight = sample_weight[:, 0]

        cl_pred = y_pred.argmax(axis=1)
        true = y_true == cl_pred

        tp = cp.zeros(y_pred.shape[1], dtype=cp.float64)
        tp.scatter_add(cl_pred, true * sample_weight)

        tot = cp.zeros(y_pred.shape[1], dtype=cp.float64)
        if mode == 'p':
            tot.scatter_add(cl_pred, sample_weight)
            return tp, tot

        tot.scatter_add(y_true, sample_weight)
        if mode == 'r':
            return tp, tot

        tot_p = cp.zeros(y_pred.shape[1], dtype=cp.float64)
        tot_p.scatter_add(cl_pred, sample_weight)

        return tp, tot, tot_p

    def get_metric(self, tp, tot):

        tot = cp.clip(tot, 1e-5, None)

        if self.average == 'micro':
            return float(tp.sum() / tot.sum())

        return float((tp / tot).mean())

    def compare(self, v0, v1):
        return v0 > v1


class MultiPrecision(MultiMetric):
    alias = 'Precision'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot = self.get_stats(y_true, y_pred, sample_weight=sample_weight, mode='p')
        return self.get_metric(tp, tot)


class MultiRecall(MultiMetric):
    alias = 'Recall'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot = self.get_stats(y_true, y_pred, sample_weight=sample_weight, mode='r')
        return self.get_metric(tp, tot)


class MultiF1Score(MultiMetric):
    alias = 'F1_score'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot, tot_p = self.get_stats(y_true, y_pred, sample_weight=sample_weight, mode='f1')
        precision = self.get_metric(tp, tot_p)
        recall = self.get_metric(tp, tot)
        return 2 * (precision * recall) / (precision + recall)


# binary metrics

def auc(y, x, sample_weight=None):
    """Roc-auc score via cupy

    Args:
        y: cp.ndarray, 1d prediction
        x: cp.ndarray, 1d binary target
        sample_weight: optional 1d array of sample weights

    Returns:

    """
    unique_x = cp.unique(x)

    if unique_x.shape[0] <= 1:
        return 0.5

    if sample_weight is None:
        sample_weight = cp.ones_like(y)

    rank_x = cp.searchsorted(unique_x, x)

    sum_1 = cp.zeros_like(unique_x, dtype=cp.float64)
    sum_1.scatter_add(rank_x, sample_weight * y)

    sum_0 = cp.zeros_like(unique_x, dtype=cp.float64)
    sum_0.scatter_add(rank_x, sample_weight * (1 - y))

    cs_0 = sum_0.cumsum()
    auc_ = (cs_0 - sum_0 / 2) * sum_1

    tot = cs_0[-1] * sum_1.sum()

    return float(auc_.sum() / tot)


class RocAucMetric(Metric):
    """Roc-auc metric for validation"""
    alias = 'AUC'

    def __call__(self, y_true, y_pred, sample_weight=None):
        """

        Args:
            y_true: cp.ndarray of targets
            y_pred: cp.ndarray of predictions
            sample_weight: None or cp.ndarray of sample_weights

        Returns:

        """
        assert y_pred.shape[1] == 1, 'Multioutput is not supported'

        if sample_weight is not None:
            sample_weight = sample_weight[:, 0]

        return auc(y_true[:, 0], y_pred[:, 0], sample_weight)

    def compare(self, v0, v1):
        return v0 > v1


class ThresholdMetrics(Metric):

    def __init__(self, threshold=0.5, q=None):
        self.threshold = threshold
        self.q = q

    def get_label(self, y_pred):
        threshold = self.threshold
        if self.q is not None:
            threshold = cp.quantile(y_pred, self.q, axis=0, interpolation='higher')

        return y_pred >= threshold

    def get_stats(self, y_true, y_pred, sample_weight=None, mode='f1'):

        y_pred = self.get_label(y_pred)
        true = y_pred == y_true

        tp = true * y_pred
        if sample_weight is not None:
            tp = tp * sample_weight
        tp = tp.sum(axis=0)

        if mode == 'p':
            if sample_weight is not None:
                return tp, (y_pred * sample_weight).sum(axis=0)
            return tp, y_pred.sum(axis=0)

        if sample_weight is not None:
            tot = (y_true * sample_weight).sum(axis=0)
        else:
            tot = y_true.sum(axis=0)
        if mode == 'r':
            return tp, tot

        if sample_weight is not None:
            tot_p = (y_pred * sample_weight).sum(axis=0)
        else:
            tot_p = y_pred.sum(axis=0)

        return tp, tot, tot_p

    def compare(self, v0, v1):
        return v0 > v1


class AccuracyMetric(ThresholdMetrics):
    alias = 'Accuracy'

    def error(self, y_true, y_pred):
        y_pred = self.get_label(y_pred)
        return (y_true == y_pred).mean(axis=1)


class Precision(ThresholdMetrics):
    alias = 'Precision'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot = self.get_stats(y_true, y_pred, sample_weight, mode='p')
        tot = cp.clip(tot, 1e-5, None)
        return (tp / tot).mean()


class Recall(ThresholdMetrics):
    alias = 'Recall'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot = self.get_stats(y_true, y_pred, sample_weight, mode='r')
        tot = cp.clip(tot, 1e-5, None)
        return (tp / tot).mean()


class F1Score(ThresholdMetrics):
    alias = 'F1_score'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot, tot_p = self.get_stats(y_true, y_pred, sample_weight, mode='f1')
        precision = tp / cp.clip(tot_p, 1e-5, None)
        recall = tp / cp.clip(tot, 1e-5, None)

        return (2 * (precision * recall) / cp.clip(precision + recall, 1e-5, None)).mean()


loss_alias = {

    # for bce
    'binary': BCELoss(),
    'bce': BCELoss(),
    'multilabel': BCELoss(),
    'logloss': BCELoss(),

    # for multiclass
    'multiclass': CrossEntropyLoss(),
    'crossentropy': CrossEntropyLoss(),

    # for regression
    'mse': MSELoss(),
    'regression': MSELoss(),
    'l2': MSELoss(),
    'multitask': MSELoss(),

}

metric_alias = {

    # for bce
    'bce': BCEMetric(),
    'logloss': BCEMetric(),

    # for multiclass
    'crossentropy': CrossEntropyMetric(),

    # for regression
    'rmse': RMSEMetric(),
    'l2': RMSEMetric(),

}

multiclass_metric_alias = {

}
