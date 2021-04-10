# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
# cython: initializedcheck=False, nonecheck=False

"""Teaching Learning Based Optimization

author: Yuan Chang
copyright: Copyright (C) 2016-2021
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from libc.math cimport round
from numpy import zeros, float64 as f64
from .utility cimport uint, rand_v, rand_i, ObjFunc, Algorithm


@cython.final
cdef class TLBO(Algorithm):
    """The implementation of Teaching Learning Based Optimization."""
    cdef double[:] tmp

    def __cinit__(
        self,
        ObjFunc func not None,
        dict settings not None,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        self.tmp = zeros(self.dim, dtype=f64)

    cdef inline void bounding(self, uint s) nogil:
        if self.tmp[s] < self.func.lb[s]:
            self.tmp[s] = self.func.lb[s]
        elif self.tmp[s] > self.func.ub[s]:
            self.tmp[s] = self.func.ub[s]

    cdef inline void register(self, uint i) nogil:
        cdef double f_new = self.func.fitness(self.tmp)
        if f_new < self.fitness[i]:
            self.pool[i, :] = self.tmp
            self.fitness[i] = f_new
        if self.fitness[i] < self.best_f:
            self.set_best(i)

    cdef inline void teaching(self, uint i) nogil:
        """Teaching phase. The last best is the teacher."""
        cdef double tf = round(1 + rand_v())
        cdef uint s, j
        cdef double mean
        for s in range(self.dim):
            mean = 0
            for j in range(self.pop_num):
                mean += self.pool[j, s]
            mean /= self.dim
            self.tmp[s] = self.pool[i, s] + rand_v(1, self.dim) * (
                self.best[s] - tf * mean)
            self.bounding(s)
        self.register(i)

    cdef inline void learning(self, uint i) nogil:
        """Learning phase."""
        cdef uint j = rand_i(self.pop_num - 1)
        if j >= i:
            j += 1
        cdef uint s
        cdef double diff
        for s in range(self.dim):
            if self.fitness[j] < self.fitness[i]:
                diff = self.pool[i, s] - self.pool[j, s]
            else:
                diff = self.pool[j, s] - self.pool[i, s]
            self.tmp[s] = self.pool[i, s] + diff * rand_v(1, self.dim)
            self.bounding(s)
        self.register(i)

    cdef inline void generation(self) nogil:
        """The process of each generation."""
        cdef uint i
        for i in range(self.pop_num):
            self.teaching(i)
            self.learning(i)
