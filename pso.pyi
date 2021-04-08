# -*- coding: utf-8 -*-

from typing import Callable, Optional
from .utility import Algorithm, ObjFunc, FVal
from .config_types import PSOSetting

class PSO(Algorithm):

    def __init__(
        self,
        func: ObjFunc[FVal],
        settings: PSOSetting,
        progress_fun: Optional[Callable[[int, str], None]] = None,
        interrupt_fun: Optional[Callable[[], bool]] = None
    ):
        """The format of argument `settings`:

        + `pop_num`: Population
            + type: int
            + default: 500
        + `cognition`: Cognition rate
            + type: float (1.~)
            + default: 2.05
        + `social`: Social rate
            + type: float (1.~)
            + default: 2.05
        + `velocity`: Velocity rate
            + type: float (1.~)
            + default: 1.3
        + `max_gen` or `min_fit` or `max_time`: Limitation of termination
            + type: int / float / float
            + default: Raise `ValueError`
        + `report`: Report per generation
            + type: int
            + default: 10

        Others arguments are same as [`Differential.__init__()`](#differential9595init__).
        """
        ...
