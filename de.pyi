# -*- coding: utf-8 -*-

from typing import Callable, Optional
from .utility import Algorithm, ObjFunc, FVal
from .config_types import DEConfig

class Differential(Algorithm):

    def __init__(
        self,
        func: ObjFunc[FVal],
        settings: DEConfig,
        progress_fun: Optional[Callable[[int, str], None]] = None,
        interrupt_fun: Optional[Callable[[], bool]] = None
    ):
        """The argument `func` is a object inherit from [Verification],
        and all abstract methods should be implemented.

        The format of argument `settings`:

        + `strategy`: Strategy
            + type: int (0~9)
            + default: 0
        + `pop_num`: Population
            + type: int
            + default: 400
        + `F`: Weight factor
            + type: float (0.~1.)
            + default: 0.6
        + `CR`: Crossover rate
            + type: float (0.~1.)
            + default: 0.9
        + `max_gen` or `min_fit` or `max_time` or `slow_down`: Limitation of termination
            + type: int / float / float / float
            + default: Raise `ValueError`
        + `report`: Report per generation
            + type: int
            + default: 10

        !!! note
            The option `slow_down` is a percent value that
            current fitness difference of two generation is divide by last one.

        The argument `progress_fun` will be called when update progress,
        and the argument `interrupt_fun` will check the interrupt status from GUI or subprocess.
        """
        ...
