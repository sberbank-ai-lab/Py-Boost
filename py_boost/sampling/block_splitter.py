import cupy as cp

from ..callbacks.callback import Callback


class SingleSplitter(Callback):

    def __init__(self):
        self.ensemble_indexer = None
        self.indexer = None

    def before_iteration(self, build_info):
        if build_info['num_iter'] == 0:
            nout = build_info['data']['train']['grad'].shape[1]
            self.indexer = cp.arange(nout, dtype=cp.uint64)

    def __call__(self):
        return [self.indexer]

    def after_train(self, build_info):
        self.__init__()


class RandomGroupsSplitter(SingleSplitter):

    def __init__(self, ngroups=2):
        super().__init__()
        self.ngroups = ngroups
        self._ngroups = None

    def before_iteration(self, build_info):
        super().before_iteration(build_info)
        if build_info['num_iter'] == 0:
            self._ngroups = min(self.ngroups, build_info['data']['train']['grad'].shape[1])

    def __call__(self):
        cp.random.shuffle(self.indexer)
        return cp.array_split(self.indexer, self._ngroups)


class OneVsAllSplitter(SingleSplitter):

    def __call__(self):
        return cp.array_split(self.indexer, self.indexer.shape[0])
