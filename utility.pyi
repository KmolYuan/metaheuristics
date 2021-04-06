# -*- coding: utf-8 -*-

from abc import abstractmethod
from typing import TypeVar, Tuple, Sequence, Callable, Optional, Generic
from numpy import ndarray, double
from .config_types import Setting

FVal = TypeVar('FVal')

class ObjFunc(Generic[FVal]):

    @abstractmethod
    def fitness(self, v: ndarray) -> double:
        """(`cdef` function) Return the fitness from the variable list `v`.
        This function will be directly called in the algorithms.
        """
        ...

    def result(self, v: ndarray) -> FVal:
        ...

class Algorithm(Generic[FVal]):
    func: ObjFunc[FVal]

    def __class_getitem__(cls, item):
        # PEP 560
        raise NotImplemented

    @abstractmethod
    def __init__(
        self,
        func: ObjFunc[FVal],
        settings: Setting,
        progress_fun: Optional[Callable[[int, str], None]] = None,
        interrupt_fun: Optional[Callable[[], bool]] = None
    ):
        """The argument `func` is an object inherit from [ObjFunc],
        and all abstract methods should be implemented.

        The format of argument `settings` can be customized.

        The argument `progress_fun` will be called when update progress,
        and the argument `interrupt_fun` will check the interrupt status from
        GUI or subprocess.
        """
        ...

    def history(self) -> ndarray:
        ...

    def result(self) -> Tuple[ndarray, float]:
        ...

    def run(self) -> FVal:
        ...
