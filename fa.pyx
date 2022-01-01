# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
# cython: initializedcheck=False, nonecheck=False

"""Firefly Algorithm

author: Yuan Chang
copyright: Copyright (C) 2016-2022
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from libc.math cimport exp
from numpy import zeros, float64 as f64
from .utility cimport uint, rand_v, ObjFunc, Algorithm


cdef double _distance(double[:] me, double[:] she, uint dim) nogil:
    """Distance of two fireflies."""
    cdef double dist = 0
    cdef uint i
    cdef double diff
    for i in range(dim):
        diff = me[i] - she[i]
        dist += diff * diff
    return dist


@cython.final
cdef class FA(Algorithm):
    """The implementation of Firefly Algorithm."""
    cdef double alpha, beta_min, gamma

    def __cinit__(
        self,
        ObjFunc func not None,
        dict settings not None,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        # alpha, the step size
        self.alpha = settings['alpha']
        # beta_min, the minimal attraction, must not less than this
        self.beta_min = settings['beta_min']
        # gamma
        self.gamma = settings['gamma']

    cdef inline void move_fireflies(self) nogil:
        """Move fireflies."""
        cdef uint i, j
        for i in range(self.pop_num - 1):
            for j in range(i + 1, self.pop_num):
                self.move_firefly(i, j)

    cdef inline void move_firefly(self, uint i, uint j) nogil:
        """Move single firefly."""
        if self.fitness[i] <= self.fitness[j]:
            i, j = j, i
        cdef double[:] v
        with gil:
            v = zeros(self.dim, dtype=f64)
        cdef double r = _distance(self.pool[i, :], self.pool[j, :], self.dim)
        cdef double beta = self.beta_min * exp(-self.gamma * r)
        cdef uint s
        cdef double step, surround
        for s in range(self.dim):
            step = self.alpha * (self.func.ub[s] - self.func.lb[s]) * rand_v(-0.5, 0.5)
            surround = self.pool[i, s] + beta * (self.pool[j, s] - self.pool[i, s])
            v[s] = self.check(s, surround + step)
        cdef double f = self.func.fitness(v)
        if f < self.fitness[i]:
            self.assign_from(i, f, v)

    cdef inline void generation(self) nogil:
        self.move_fireflies()
        self.alpha *= 0.95
        self.find_best()
