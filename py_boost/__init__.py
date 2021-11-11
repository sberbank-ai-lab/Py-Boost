import sys
import logging

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.WARNING)

# if root logger has handlers, propagate messages up and let root logger process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler(sys.stdout))
    _logger.propagate = False

from .gpu.boosting import GradientBoosting

__version__ = '0.1.0'

__all__ = [

    'GradientBoosting',
    'callbacks',
    'gpu',
    'multioutput',
    'sampling',
    'utils'

]

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)