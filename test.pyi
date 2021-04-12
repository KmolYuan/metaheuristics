# -*- coding: utf-8 -*-

from numpy import ndarray, double
from .utility import ObjFunc


class TestObj(ObjFunc[float]):

    def fitness(self, v: ndarray) -> double:
        ...

    def result(self, v: ndarray) -> float:
        ...
