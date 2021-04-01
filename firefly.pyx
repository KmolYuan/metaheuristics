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
from cython.parallel cimport prange
from libc.math cimport exp, sqrt
from .utility cimport uint, rand_v, ObjFunc, Algorithm


cdef double _distance(double[:] me, double[:] she, uint dim) nogil:
    """Distance of two fireflies."""
    cdef double dist = 0
    cdef uint i
    cdef double diff
    for i in range(dim):
        diff = me[i] - she[i]
        dist += diff * diff
    return sqrt(dist)


@cython.final
cdef class Firefly(Algorithm):
    """The implementation of Firefly Algorithm."""
    cdef double alpha, beta_min, beta0, gamma

    def __cinit__(
        self,
        ObjFunc func not None,
        dict settings not None,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        """
        settings = {
            'pop_num': int,
            'alpha': float,
            'beta_min': float,
            'beta0': float,
            'gamma': float,
            'max_gen': int or 'min_fit': float or 'max_time': float,
            'report': int,
        }
        """
        # n, the population size of fireflies
        self.pop_num = settings.get('pop_num', 80)
        # alpha, the step size
        self.alpha = settings.get('alpha', 0.01)
        # beta_min, the minimal attraction, must not less than this
        self.beta_min = settings.get('beta_min', 0.2)
        # beta0, the attraction of two firefly in 0 distance
        self.beta0 = settings.get('beta0', 1.)
        # gamma
        self.gamma = settings.get('gamma', 1.)
        # all fireflies, depended on population n
        self.new_pop()

    cdef inline void initialize(self) nogil:
        """Initial population."""
        self.initialize_pop()
        self.set_best(0)

    cdef inline void move_fireflies(self) nogil:
        """Move fireflies."""
        cdef bint is_move
        cdef uint i, j, s
        cdef double scale, tmp_v
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
                            * exp(-self.gamma * r * r) + self.beta_min)
        cdef uint s
        for s in range(self.dim):
            me[s] = self.check(s, me[s] + beta * (she[s] - me[s]) + self.alpha
                               * (self.func.ub[s] - self.func.lb[s])
                               * rand_v(-0.5, 0.5))

    cdef inline double check(self, int s, double v) nogil:
        """Check the bounds."""
        if v > self.func.ub[s]:
            return self.func.ub[s]
        elif v < self.func.lb[s]:
            return self.func.lb[s]
        else:
            return v

    cdef inline void generation_process(self) nogil:
        self.move_fireflies()
        # Get fitness
        cdef uint i
        if self.parallel:
            for i in prange(self.pop_num):
                self.fitness[i] = self.func.fitness(self.pool[i, :])
        else:
            for i in range(self.pop_num):
                self.fitness[i] = self.func.fitness(self.pool[i, :])
        self.find_best()
