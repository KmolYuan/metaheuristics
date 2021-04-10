# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
# cython: initializedcheck=False, nonecheck=False

"""Firefly Algorithm

author: Yuan Chang
copyright: Copyright (C) 2016-2021
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from libc.math cimport exp
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
    cdef double alpha, beta_min, beta0, gamma

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
        # beta0, the attraction of two firefly in 0 distance
        self.beta0 = settings['beta0']
        # gamma
        self.gamma = settings['gamma']

    cdef inline void move_fireflies(self) nogil:
        """Move fireflies."""
        cdef bint is_move
        cdef uint i, j, s
        for i in range(self.pop_num):
            moved = False
            for j in range(self.pop_num):
                if i == j or self.fitness[i] <= self.fitness[j]:
                    continue
                self.move_firefly(self.pool[i, :], self.pool[j, :])
                moved = True
            if moved:
                continue
            # Evaluate
            for s in range(self.dim):
                self.pool[i, s] = self.check(s, self.pool[i, s] + self.alpha * (
                    self.func.ub[s] - self.func.lb[s]) * rand_v(-0.5, 0.5))

    cdef inline void move_firefly(self, double[:] me, double[:] she) nogil:
        """Move single firefly."""
        cdef double r = _distance(me, she, self.dim)
        cdef double beta = ((self.beta0 - self.beta_min)
                            * exp(-self.gamma * r) + self.beta_min)
        cdef uint s
        for s in range(self.dim):
            me[s] = self.check(s, me[s] + beta * (she[s] - me[s]) + self.alpha
                               * (self.func.ub[s] - self.func.lb[s])
                               * rand_v(-0.5, 0.5))

    cdef inline void generation(self) nogil:
        self.move_fireflies()
        # Get fitness
        cdef uint i
        for i in range(self.pop_num):
            self.fitness[i] = self.func.fitness(self.pool[i, :])
        self.find_best()
