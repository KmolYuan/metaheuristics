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
from cython.parallel cimport parallel
from libc.math cimport pow
from numpy import zeros, float64 as f64
from .utility cimport uint, MAX_GEN, rand_v, rand_i, ObjFunc, Algorithm


@cython.final
cdef class Genetic(Algorithm):
    """The implementation of Real-coded Genetic Algorithm."""
    cdef double cross, mutate_f, win, delta
    cdef double[:] new_fitness, tmp1, tmp2, tmp3, f_tmp
    cdef double[:, :] new_pool

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
            'cross': float,
            'mutate': float,
            'win': float,
            'delta': float,
            'max_gen': int or 'min_fit': float or 'max_time': float,
            'report': int,
        }
        """
        self.cross = settings.get('cross', 0.95)
        self.mutate_f = settings.get('mutate', 0.05)
        self.win = settings.get('win', 0.95)
        self.delta = settings.get('delta', 5.)
        self.new_fitness = zeros(self.pop_num, dtype=f64)
        self.new_pool = zeros((self.pop_num, self.dim), dtype=f64)
        self.tmp1 = self.make_tmp()
        self.tmp2 = self.make_tmp()
        self.tmp3 = self.make_tmp()
        self.f_tmp = zeros(3, dtype=f64)

    cdef inline double check(self, int i, double v) nogil:
        """If a variable is out of bound, replace it with a random value."""
        if not self.func.ub[i] >= v >= self.func.lb[i]:
            return rand_v(self.func.lb[i], self.func.ub[i])
        return v

    cdef inline void init(self) nogil:
        self.init_pop()
        self.set_best(0)

    cdef inline void crossover(self) nogil:
        cdef uint i, s
        for i in range(0, self.pop_num - 1, 2):
            if not rand_v() < self.cross:
                continue
            for s in range(self.dim):
                # first baby, half father half mother
                self.tmp1[s] = 0.5 * self.pool[i, s] + 0.5 * self.pool[i + 1, s]
                # second baby, three quarters of father and quarter of mother
                self.tmp2[s] = self.check(s, 1.5 * self.pool[i, s]
                                   - 0.5 * self.pool[i + 1, s])
                # third baby, quarter of father and three quarters of mother
                self.tmp3[s] = self.check(s, -0.5 * self.pool[i, s]
                                   + 1.5 * self.pool[i + 1, s])
            # evaluate new baby
            self.f_tmp[0] = self.func.fitness(self.tmp1)
            self.f_tmp[1] = self.func.fitness(self.tmp2)
            self.f_tmp[2] = self.func.fitness(self.tmp3)
            # bubble sort: smaller -> larger
            if self.f_tmp[0] > self.f_tmp[1]:
                self.f_tmp[0], self.f_tmp[1] = self.f_tmp[1], self.f_tmp[0]
                self.tmp1, self.tmp2 = self.tmp2, self.tmp1
            if self.f_tmp[0] > self.f_tmp[2]:
                self.f_tmp[0], self.f_tmp[2] = self.f_tmp[2], self.f_tmp[0]
                self.tmp1, self.tmp3 = self.tmp3, self.tmp1
            if self.f_tmp[1] > self.f_tmp[2]:
                self.f_tmp[1], self.f_tmp[2] = self.f_tmp[2], self.f_tmp[1]
                self.tmp2, self.tmp3 = self.tmp3, self.tmp2
            # replace first two baby to parent, another one will be
            self.assign_from(i, self.f_tmp[0], self.tmp1)
            self.assign_from(i + 1, self.f_tmp[1], self.tmp2)

    cdef inline double get_delta(self, double y) nogil:
        cdef double r
        if self.stop_at == MAX_GEN and self.stop_at_i > 0:
            r = <double>self.func.gen / self.stop_at_i
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
            if self.parallel:
                with parallel():
                    self.fitness[i] = self.func.fitness(self.pool[i, :])
            else:
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
        # select random one chromosome to be best chromosome, make best chromosome still exist
        self.assign_from(rand_i(self.pop_num), self.best_f, self.best)

    cdef inline void generation(self) nogil:
        self.select()
        self.crossover()
        self.mutate()
