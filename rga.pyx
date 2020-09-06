# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True

"""Real-coded Genetic Algorithm

author: Yuan Chang
copyright: Copyright (C) 2016-2020
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from libc.math cimport pow
from numpy import zeros, float64 as np_float
from .utility cimport MAX_GEN, rand_v, rand_i, ObjFunc, Algorithm

ctypedef unsigned int uint


@cython.final
cdef class Genetic(Algorithm):
    """The implementation of Real-coded Genetic Algorithm."""
    cdef double cross, mute, win, delta
    cdef double[:] new_fitness
    cdef double[:, :] new_pool

    def __cinit__(
        self,
        ObjFunc func,
        dict settings,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        """
        settings = {
            'pop_num': int,
            'cross': float,
            'mute': float,
            'win': float,
            'delta': float,
            'max_gen': int or 'min_fit': float or 'max_time': float,
            'report': int,
        }
        """
        self.pop_num = settings.get('pop_num', 500)
        self.cross = settings.get('cross', 0.95)
        self.mute = settings.get('mute', 0.05)
        self.win = settings.get('win', 0.95)
        self.delta = settings.get('delta', 5.)
        self.new_pop()
        self.new_fitness = zeros(self.pop_num, dtype=np_float)
        self.new_pool = zeros((self.pop_num, self.dim), dtype=np_float)

    cdef inline void backup(self, uint i, uint j) nogil:
        """Backup individual."""
        self.new_fitness[i] = self.fitness[j]
        self.new_pool[i, :] = self.pool[j, :]

    cdef inline double check(self, int i, double v) nogil:
        """If a variable is out of bound, replace it with a random value."""
        if not self.func.ub[i] >= v >= self.func.lb[i]:
            return rand_v(self.func.lb[i], self.func.ub[i])
        return v

    cdef inline void initialize(self):
        cdef uint i, j
        for i in range(self.pop_num):
            for j in range(self.dim):
                self.pool[i, j] = rand_v(self.func.lb[j], self.func.ub[j])
        self.fitness[0] = self.func.fitness(self.pool[0, :])
        self.set_best(0)
        self.get_fitness()

    cdef inline void cross_over(self):
        cdef double[:] c1 = self.make_tmp()
        cdef double[:] c2 = self.make_tmp()
        cdef double[:] c3 = self.make_tmp()
        cdef uint i, s
        cdef double c1_f, c2_f, c3_f
        for i in range(0, self.pop_num - 1, 2):
            if not rand_v() < self.cross:
                continue
            for s in range(self.dim):
                # first baby, half father half mother
                c1[s] = 0.5 * self.pool[i, s] + 0.5 * self.pool[i + 1, s]
                # second baby, three quarters of father and quarter of mother
                c2[s] = self.check(s, 1.5 * self.pool[i, s]
                                   - 0.5 * self.pool[i + 1, s])
                # third baby, quarter of father and three quarters of mother
                c3[s] = self.check(s, -0.5 * self.pool[i, s]
                                   + 1.5 * self.pool[i + 1, s])
            # evaluate new baby
            c1_f = self.func.fitness(c1)
            c2_f = self.func.fitness(c2)
            c3_f = self.func.fitness(c3)
            # bubble sort: smaller -> larger
            if c1_f > c2_f:
                c1_f, c2_f = c2_f, c1_f
                c1, c2 = c2, c1
            if c1_f > c3_f:
                c1_f, c3_f = c3_f, c1_f
                c1, c3 = c3, c1
            if c2_f > c3_f:
                c2_f, c3_f = c3_f, c2_f
                c2, c3 = c3, c2
            # replace first two baby to parent, another one will be
            self.assign_from(i, c1_f, c1)
            self.assign_from(i + 1, c2_f, c2)

    cdef inline double get_delta(self, double y) nogil:
        cdef double r
        if self.stop_at == MAX_GEN and self.stop_at_i > 0:
            r = <double>self.func.gen / self.stop_at_i
        else:
            r = 1
        return y * rand_v() * pow(1.0 - r, self.delta)

    cdef inline void get_fitness(self):
        cdef uint i
        for i in range(self.pop_num):
            self.fitness[i] = self.func.fitness(self.pool[i, :])
        cdef uint best = 0
        for i in range(1, self.pop_num):
            if self.fitness[i] < self.fitness[best]:
                best = i
        if self.fitness[best] < self.best_f:
            self.set_best(best)

    cdef inline void mutate(self) nogil:
        cdef uint i, s
        for i in range(self.pop_num):
            if not rand_v() < self.mute:
                continue
            s = rand_i(self.dim)
            if rand_v() < 0.5:
                self.pool[i, s] += self.get_delta(self.func.ub[s]
                                                  - self.pool[i, s])
            else:
                self.pool[i, s] -= self.get_delta(self.pool[i, s]
                                                  - self.func.lb[s])

    cdef inline void select(self) nogil:
        """roulette wheel selection"""
        cdef uint i, j, k
        for i in range(self.pop_num):
            j = rand_i(self.pop_num)
            k = rand_i(self.pop_num)
            if self.fitness[j] > self.fitness[k] and rand_v() < self.win:
                self.backup(i, k)
            else:
                self.backup(i, j)
        # in this stage, new_chromosome is select finish
        # now replace origin chromosome
        for i in range(self.pop_num):
            self.fitness[:] = self.new_fitness
            self.pool[:] = self.new_pool
        # select random one chromosome to be best chromosome, make best chromosome still exist
        self.assign_from(rand_i(self.pop_num), self.best_f, self.best)

    cdef inline void generation_process(self):
        self.select()
        self.cross_over()
        self.mutate()
        self.get_fitness()
