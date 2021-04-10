# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
# cython: initializedcheck=False, nonecheck=False

"""Real-coded Genetic Algorithm

author: Yuan Chang
copyright: Copyright (C) 2016-2021
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from libc.math cimport pow
from numpy import zeros, float64 as f64
from .utility cimport uint, MAX_GEN, rand_v, rand_i, ObjFunc, Algorithm


@cython.final
cdef class RGA(Algorithm):
    """The implementation of Real-coded Genetic Algorithm."""
    cdef double cross, mutate_f, win, delta
    cdef double[:] new_fitness, f_tmp
    cdef double[:, :] new_pool, tmp

    def __cinit__(
        self,
        ObjFunc func not None,
        dict settings not None,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        self.cross = settings['cross']
        self.mutate_f = settings['mutate']
        self.win = settings['win']
        self.delta = settings['delta']
        self.new_fitness = zeros(self.pop_num, dtype=f64)
        self.new_pool = zeros((self.pop_num, self.dim), dtype=f64)
        self.tmp = zeros((3, self.dim), dtype=f64)
        self.f_tmp = zeros(3, dtype=f64)

    cdef inline double bound(self, int i, double v) nogil:
        """If a variable is out of bound, replace it with a random value."""
        if not self.func.ub[i] >= v >= self.func.lb[i]:
            return rand_v(self.func.lb[i], self.func.ub[i])
        return v

    cdef inline void crossover(self) nogil:
        cdef uint i, s
        for i in range(0, self.pop_num - 1, 2):
            if not rand_v() < self.cross:
                continue
            for s in range(self.dim):
                # first baby, half father half mother
                self.tmp[0, s] = 0.5 * (self.pool[i, s] + self.pool[i + 1, s])
                # second baby, three quarters of father and quarter of mother
                self.tmp[1, s] = self.bound(s, 1.5 * self.pool[i, s]
                                            - 0.5 * self.pool[i + 1, s])
                # third baby, quarter of father and three quarters of mother
                self.tmp[2, s] = self.bound(s, -0.5 * self.pool[i, s]
                                            + 1.5 * self.pool[i + 1, s])
            # evaluate new baby
            self.f_tmp[0] = self.func.fitness(self.tmp[0, :])
            self.f_tmp[1] = self.func.fitness(self.tmp[1, :])
            self.f_tmp[2] = self.func.fitness(self.tmp[2, :])
            # bubble sort: smaller -> larger
            if self.f_tmp[0] > self.f_tmp[1]:
                self.f_tmp[0], self.f_tmp[1] = self.f_tmp[1], self.f_tmp[0]
                self.tmp[0], self.tmp[1] = self.tmp[1], self.tmp[0]
            if self.f_tmp[0] > self.f_tmp[2]:
                self.f_tmp[0], self.f_tmp[2] = self.f_tmp[2], self.f_tmp[0]
                self.tmp[0], self.tmp[2] = self.tmp[2], self.tmp[0]
            if self.f_tmp[1] > self.f_tmp[2]:
                self.f_tmp[1], self.f_tmp[2] = self.f_tmp[2], self.f_tmp[1]
                self.tmp[1], self.tmp[2] = self.tmp[2], self.tmp[1]
            # replace first two baby to parent, another one will be
            self.assign_from(i, self.f_tmp[0], self.tmp[0])
            self.assign_from(i + 1, self.f_tmp[1], self.tmp[1])

    cdef inline double get_delta(self, double y) nogil:
        cdef double r
        if self.task == MAX_GEN and self.stop_at > 0:
            r = <double>self.func.gen / self.stop_at
        else:
            r = 1
        return y * rand_v() * pow(1.0 - r, self.delta)

    cdef inline void mutate(self) nogil:
        cdef uint i, s
        for i in range(self.pop_num):
            if not rand_v() < self.mutate_f:
                continue
            s = rand_i(self.dim)
            if rand_v() < 0.5:
                self.pool[i, s] += self.get_delta(self.func.ub[s]
                                                  - self.pool[i, s])
            else:
                self.pool[i, s] -= self.get_delta(self.pool[i, s]
                                                  - self.func.lb[s])
            # Get fitness
            self.fitness[i] = self.func.fitness(self.pool[i, :])
        self.find_best()

    cdef inline void select(self) nogil:
        """roulette wheel selection"""
        cdef uint i, j, k
        for i in range(self.pop_num):
            j = rand_i(self.pop_num)
            k = rand_i(self.pop_num)
            if self.fitness[j] > self.fitness[k] and rand_v() < self.win:
                self.new_fitness[i] = self.fitness[k]
                self.new_pool[i, :] = self.pool[k, :]
            else:
                self.new_fitness[i] = self.fitness[j]
                self.new_pool[i, :] = self.pool[j, :]
        # in this stage, new_chromosome is select finish
        # now replace origin chromosome
        self.fitness[:] = self.new_fitness
        self.pool[:] = self.new_pool
        # select random one chromosome to be best chromosome,
        # make best chromosome still exist
        self.assign_from(rand_i(self.pop_num), self.best_f, self.best)

    cdef inline void generation(self) nogil:
        self.select()
        self.crossover()
        self.mutate()
