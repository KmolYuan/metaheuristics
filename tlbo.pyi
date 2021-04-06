# -*- coding: utf-8 -*-

from typing import Callable, Optional
from .utility import Algorithm, ObjFunc, FVal
from .config_types import TOBLSetting

class TLBO(Algorithm):

    def __init__(
        self,
        func: ObjFunc[FVal],
        settings: TOBLSetting,
        progress_fun: Optional[Callable[[int, str], None]] = None,
        interrupt_fun: Optional[Callable[[], bool]] = None
    ):
        """The format of argument `settings`:

        + `pop_num`: The number of students per class
            + type: int
            + default: 50
        + `max_gen` or `min_fit` or `max_time`: Limitation of termination
            + type: int / float / float
            + default: Raise `ValueError`
        + `report`: Report per generation
            + type: int
            + default: 10

        Others arguments are same as [`Differential.__init__()`](#differential9595init__).
        """
        ...
