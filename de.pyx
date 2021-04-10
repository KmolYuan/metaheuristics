# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
# cython: initializedcheck=False, nonecheck=False

"""Differential Evolution

author: Yuan Chang
copyright: Copyright (C) 2016-2021
license: AGPL
email: pyslvs@gmail.com
"""

cimport cython
from numpy import zeros, uint32 as usize, float64 as f64
from .utility cimport uint, rand_v, rand_i, ObjFunc, Algorithm


cpdef enum Strategy:
    S1
    S2
    S3
    S4
    S5
    S6
    S7
    S8
    S9
    S10


@cython.final
cdef class DE(Algorithm):
    """The implementation of Differential Evolution."""
    cdef Strategy strategy
    cdef double f, cr
    cdef uint[:] v
    cdef double[:] tmp
    cdef (double (*)(DE, uint) nogil) formula
    cdef (void (*)(DE, uint) nogil) setter

    def __cinit__(
        self,
        ObjFunc func not None,
        dict settings not None,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        # strategy 0~9, choice what strategy to generate new member in temporary
        self.strategy = Strategy(settings['strategy'])
        # weight factor f is usually between 0.5 and 1 (in rare cases > 1)
        self.f = settings['f']
        if 0.5 > self.f or self.f > 1:
            raise ValueError('cr should be [0.5,1]')
        # crossover possible cr in [0,1]
        self.cr = settings['cr']
        if 0 > self.cr or self.cr > 1:
            raise ValueError('cr should be [0,1]')
        # the vector
        cdef uint num
        if self.strategy in {S1, S3, S6, S8}:
            num = 2
        elif self.strategy in {S2, S7}:
            num = 3
        elif self.strategy in {S4, S9}:
            num = 4
        else:
            num = 5
        self.v = zeros(num, dtype=usize)
        self.tmp = zeros(self.dim, dtype=f64)
        if self.strategy in {S1, S6}:
            self.formula = DE.f1
        elif self.strategy in {S2, S7}:
            self.formula = DE.f2
        elif self.strategy in {S3, S8}:
            self.formula = DE.f3
        elif self.strategy in {S4, S9}:
            self.formula = DE.f4
        else:
            self.formula = DE.f5
        if self.strategy in {S1, S2, S3, S4, S5}:
            self.setter = DE.s1
        else:
            self.setter = DE.s2

    cdef inline void init(self) nogil:
        """Initial population."""
        self.init_pop()
        self.find_best()

    cdef inline void vector(self, uint i) nogil:
        """Generate new vectors."""
        cdef uint j
        for j in range(len(self.v)):
            self.v[j] = i
            while True:
                if self.v[j] != i:
                    for k in range(j):
                        if self.v[j] == self.v[k]:
                            break
                    else:
                        break
                self.v[j] = rand_i(self.pop_num)

    cdef double f1(self, uint n) nogil:
        return self.best[n] + self.f * (
            self.pool[self.v[0], n] - self.pool[self.v[1], n])

    cdef double f2(self, uint n) nogil:
        return self.pool[self.v[0], n] + self.f * (
            self.pool[self.v[1], n] - self.pool[self.v[2], n])

    cdef double f3(self, uint n) nogil:
        return self.tmp[n] + self.f * (self.best[n] - self.tmp[n]
            + self.pool[self.v[0], n] - self.pool[self.v[1], n])

    cdef double f4(self, uint n) nogil:
        return self.best[n] + self.f * (
            self.pool[self.v[0], n] + self.pool[self.v[1], n]
            - self.pool[self.v[2], n] - self.pool[self.v[3], n])

    cdef double f5(self, uint n) nogil:
        return self.pool[self.v[4], n] + self.f * (
            self.pool[self.v[0], n] + self.pool[self.v[1], n]
            - self.pool[self.v[2], n] - self.pool[self.v[3], n])

    cdef void s1(self, uint n) nogil:
        for _ in range(self.dim):
            self.tmp[n] = self.formula(self, n)
            n = (n + 1) % self.dim
            if rand_v() >= self.cr:
                break

    cdef void s2(self, uint n) nogil:
        cdef uint l_v
        for l_v in range(self.dim):
            if rand_v() < self.cr or l_v == self.dim - 1:
                self.tmp[n] = self.formula(self, n)
            n = (n + 1) % self.dim

    cdef inline void recombination(self, int i) nogil:
        """use new vector, recombination the new one member to tmp."""
        self.tmp[:] = self.pool[i, :]
        self.setter(self, rand_i(self.dim))

    cdef inline void generation(self) nogil:
        cdef uint i, s
        cdef double tmp_f
        for i in range(self.pop_num):
            # Generate a new vector
            self.vector(i)
            # Use the vector recombine the member to temporary
            self.recombination(i)
            # Check the one is out of bound
            for s in range(self.dim):
                if not self.func.ub[s] >= self.tmp[s] >= self.func.lb[s]:
                    break
            else:
                # Test
                tmp_f = self.func.fitness(self.tmp)
                # Self evolution
                if tmp_f < self.fitness[i]:
                    self.assign_from(i, tmp_f, self.tmp)
        self.find_best()
