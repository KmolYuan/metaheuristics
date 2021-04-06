# -*- coding: utf-8 -*-

from typing import Callable, Optional
from .utility import Algorithm, ObjFunc, FVal
from .config_types import RGASetting

class RGA(Algorithm):

    def __init__(
        self,
        func: ObjFunc[FVal],
        settings: RGASetting,
        progress_fun: Optional[Callable[[int, str], None]] = None,
        interrupt_fun: Optional[Callable[[], bool]] = None
    ):
        """The format of argument `settings`:

        + `pop_num`: Population
            + type: int
            + default: 500
        + `cross`: Crossover rate
            + type: float (0.~1.)
            + default: 0.95
        + `mutate`: Mutation rate
            + type: float (0.~1.)
            + default: 0.05
        + `win`: Win rate
            + type: float (0.~1.)
            + default: 0.95
        + `delta`: Delta value
            + type: float
            + default: 5.
        + `max_gen` or `min_fit` or `max_time`: Limitation of termination
            + type: int / float / float
            + default: Raise `ValueError`
        + `report`: Report per generation
            + type: int
            + default: 10

        Others arguments are same as [`Differential.__init__()`](#differential9595init__).
        """
        ...
