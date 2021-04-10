# -*- coding: utf-8 -*-

"""Kernel of Metaheuristic Algorithm."""

__author__ = "Yuan Chang"
__copyright__ = "Copyright (C) 2016-2021"
__license__ = "AGPL"
__email__ = "pyslvs@gmail.com"

from typing import Mapping, Dict, Union, Type
from enum import unique, Enum
from .utility import ObjFunc, Algorithm
from .config_types import (
    Setting, DESetting, FASetting, RGASetting, TOBLSetting,
)
from .rga import RGA
from .de import DE
from .pso import PSO
from .fa import FA
from .tlbo import TLBO


@unique
class AlgorithmType(str, Enum):
    """Enum type of algorithms."""
    RGA = "Real-coded Genetic Algorithm"
    DE = "Differential Evolution"
    PSO = "Particle Swarm Optimization"
    FA = "Firefly Algorithm"
    TLBO = "Teaching Learning Based Optimization"


_ALGORITHM: Mapping[AlgorithmType, Type[Algorithm]] = {
    AlgorithmType.RGA: RGA,
    AlgorithmType.DE: DE,
    AlgorithmType.PSO: PSO,
    AlgorithmType.FA: FA,
    AlgorithmType.TLBO: TLBO,
}
_DEFAULT_PARAMS = {'max_gen': 1000, 'report': 50}
_PARAMS: Mapping[AlgorithmType, Dict[str, Union[int, float]]] = {
    AlgorithmType.RGA: {
        'pop_num': 500,
        'cross': 0.95,
        'mutate': 0.05,
        'win': 0.95,
        'delta': 5.,
    },
    AlgorithmType.DE: {
        'pop_num': 400,
        'strategy': 1,
        'f': 0.6,
        'cr': 0.9,
    },
    AlgorithmType.PSO: {
        'pop_num': 200,
        'cognition': 2.05,
        'social': 2.05,
        'velocity': 1.3,
    },
    AlgorithmType.FA: {
        'pop_num': 80,
        'alpha': 0.01,
        'beta_min': 0.2,
        'gamma': 1.,
        'beta0': 1.,
    },
    AlgorithmType.TLBO: {
        'pop_num': 50,
    },
}


def algorithm(opt: AlgorithmType) -> Type[Algorithm]:
    """Return the class of the algorithms."""
    return _ALGORITHM[opt]


def default(opt: AlgorithmType) -> Dict[str, Union[int, float]]:
    """Return the default settings of the algorithms."""
    config = _PARAMS[opt].copy()
    config.update(_DEFAULT_PARAMS.copy())
    return config
