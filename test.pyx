# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
# cython: initializedcheck=False, nonecheck=False

"""Test objective function for algorithms.

author: Yuan Chang
copyright: Copyright (C) 2016-2021
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from numpy import array, float64 as f64
from .utility cimport ObjFunc


@cython.final
cdef class TestObj(ObjFunc):
    """Test objective function.

    f(x) = x1^2 + 8*x2
    """

    def __cinit__(self):
        self.ub = array([50] * 4, dtype=f64)
        self.lb = array([0] * 4, dtype=f64)

    cdef double target(self, double[:] v) nogil:
        return v[0] * v[0] + 8 * v[1] * v[1] + v[2] * v[2] + v[3] * v[3]

    cdef double fitness(self, double[:] v) nogil:
        return self.target(v)

    cpdef object result(self, double[:] v):
        return self.target(v)
