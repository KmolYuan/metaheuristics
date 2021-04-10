# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
# cython: initializedcheck=False, nonecheck=False

"""Particle Swarm Optimization

author: Yuan Chang
copyright: Copyright (C) 2016-2021
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from numpy import zeros, float64 as f64
from .utility cimport uint, rand_v, ObjFunc, Algorithm


@cython.final
cdef class PSO(Algorithm):
    cdef double cognition, social, velocity
    cdef double[:] best_f_past
    cdef double[:, :] best_past

    def __cinit__(
        self,
        ObjFunc func not None,
        dict settings not None,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        self.cognition = settings['cognition']
        self.social = settings['social']
        self.velocity = settings['velocity']
        self.best_past = zeros((self.pop_num, self.dim), dtype=f64)
        self.best_f_past = zeros(self.pop_num, dtype=f64)

    cdef inline void init(self) nogil:
        """Initial population."""
        self.best_past[:, :] = self.pool
        self.best_f_past[:] = self.fitness

    cdef inline void generation(self) nogil:
        cdef double alpha
        cdef double beta
        cdef uint i, s
        for i in range(self.pop_num):
            alpha = self.cognition * rand_v()
            beta = self.social * rand_v()
            for s in range(self.dim):
                self.pool[i, s] = self.check(s, self.velocity * self.pool[i, s]
                    + alpha * (self.best_past[i, s] - self.pool[i, s])
                    + beta * (self.best[s] - self.pool[i, s]))
            self.fitness[i] = self.func.fitness(self.pool[i, :])
            if self.fitness[i] < self.best_f_past[i]:
                self.best_past[i, :] = self.pool[i, :]
                self.best_f_past[i] = self.fitness[i]
            if self.fitness[i] < self.best_f:
                self.set_best(i)
