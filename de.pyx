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
from .utility cimport uint, rand_v, rand_i, ObjFunc, Algorithm

ctypedef double (*F)(DE, uint) nogil


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
    cdef uint r1, r2, r3, r4, r5
    cdef double F, CR
    cdef double[:] tmp
    cdef F formula

    def __cinit__(
        self,
        ObjFunc func not None,
        dict settings not None,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        """
        settings = {
            'strategy': int,
            'pop_num': int,
            'F': float,
            'CR': float,
            'max_gen': int or 'min_fit': float or 'max_time': float,
            'report': int,
        }
        """
        # strategy 0~9, choice what strategy to generate new member in temporary
        self.strategy = Strategy(settings.get('strategy', S1))
        # weight factor F is usually between 0.5 and 1 (in rare cases > 1)
        self.F = settings.get('F', 0.6)
        if not (0.5 <= self.F <= 1):
            raise ValueError('CR should be [0.5,1]')
        # crossover possible CR in [0,1]
        self.CR = settings.get('CR', 0.9)
        if not (0 <= self.CR <= 1):
            raise ValueError('CR should be [0,1]')
        # the vector
        self.r1 = self.r2 = self.r3 = self.r4 = self.r5 = 0
        self.tmp = self.make_tmp()
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

    cdef inline void init(self) nogil:
        """Initial population."""
        self.init_pop()
        self.find_best()

    cdef inline void vector(self, uint i) nogil:
        """Generate new vectors."""
        self.r1 = self.r2 = self.r3 = self.r4 = self.r5 = i
        while self.r1 == i:
            self.r1 = rand_i(self.pop_num)
        while self.r2 in {i, self.r1}:
            self.r2 = rand_i(self.pop_num)
        if self.strategy in {S1, S3, S6, S8}:
            return
        while self.r3 in {i, self.r1, self.r2}:
            self.r3 = rand_i(self.pop_num)
        if self.strategy in {S2, S7}:
            return
        while self.r4 in {i, self.r1, self.r2, self.r3}:
            self.r4 = rand_i(self.pop_num)
        if self.strategy in {S4, S9}:
            return
        while self.r5 in {i, self.r1, self.r2, self.r3, self.r4}:
            self.r5 = rand_i(self.pop_num)

    cdef double f1(self, uint n) nogil:
        return self.best[n] + self.F * (
            self.pool[self.r1, n] - self.pool[self.r2, n])

    cdef double f2(self, uint n) nogil:
        return self.pool[self.r1, n] + self.F * (
            self.pool[self.r2, n] - self.pool[self.r3, n])

    cdef double f3(self, uint n) nogil:
        return self.tmp[n] + self.F * (self.best[n] - self.tmp[n]
            + self.pool[self.r1, n] - self.pool[self.r2, n])

    cdef double f4(self, uint n) nogil:
        return self.best[n] + self.F * (
            self.pool[self.r1, n] + self.pool[self.r2, n]
            - self.pool[self.r3, n] - self.pool[self.r4, n])

    cdef double f5(self, uint n) nogil:
        return self.pool[self.r5, n] + self.F * (
            self.pool[self.r1, n] + self.pool[self.r2, n]
            - self.pool[self.r3, n] - self.pool[self.r4, n])

    cdef inline void recombination(self, int i) nogil:
        """use new vector, recombination the new one member to tmp."""
        self.tmp[:] = self.pool[i, :]
        cdef uint n = rand_i(self.dim)
        cdef uint l_v
        if self.strategy in {S1, S2, S3, S4, S5}:
            l_v = 0
            while True:
                self.tmp[n] = self.formula(self, n)
                n = (n + 1) % self.dim
                l_v += 1
                if rand_v() >= self.CR or l_v >= self.dim:
                    break
        else:
            for l_v in range(self.dim):
                if rand_v() < self.CR or l_v == self.dim - 1:
                    self.tmp[n] = self.formula(self, n)
                n = (n + 1) % self.dim

    cdef inline bint check(self) nogil:
        """check the member's chromosome that is out of bound?"""
        cdef uint i
        for i in range(self.dim):
            if not self.func.ub[i] >= self.tmp[i] >= self.func.lb[i]:
                return True
        return False

    cdef inline void generation(self) nogil:
        cdef uint i
        cdef double tmp_f
        for i in range(self.pop_num):
            # Generate a new vector
            self.vector(i)
            # Use the vector recombine the member to temporary
            self.recombination(i)
            # Check the one is out of bound
            if self.check():
                continue
            # Test
            tmp_f = self.func.fitness(self.tmp)
            # Self evolution
            if tmp_f < self.fitness[i]:
                self.assign_from(i, tmp_f, self.tmp)
        self.find_best()
