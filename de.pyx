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

ctypedef void (*Eq)(Differential, uint) nogil


cdef enum Strategy:
    STRATEGY0
    STRATEGY1
    STRATEGY2
    STRATEGY3
    STRATEGY4
    STRATEGY5
    STRATEGY6
    STRATEGY7
    STRATEGY8
    STRATEGY9


@cython.final
cdef class Differential(Algorithm):
    """The implementation of Differential Evolution."""
    cdef Strategy strategy
    cdef uint r1, r2, r3, r4, r5
    cdef double F, CR
    cdef double[:] tmp

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
        cdef uint strategy = settings.get('strategy', STRATEGY1)
        if strategy > 9:
            raise ValueError(f"invalid strategy: {strategy}")
        self.strategy = <Strategy>strategy
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
        if self.strategy in {STRATEGY1, STRATEGY3, STRATEGY6, STRATEGY8}:
            return
        while self.r3 in {i, self.r1, self.r2}:
            self.r3 = rand_i(self.pop_num)
        if self.strategy in {STRATEGY2, STRATEGY7}:
            return
        while self.r4 in {i, self.r1, self.r2, self.r3}:
            self.r4 = rand_i(self.pop_num)
        if self.strategy in {STRATEGY4, STRATEGY9}:
            return
        while self.r5 in {i, self.r1, self.r2, self.r3, self.r4}:
            self.r5 = rand_i(self.pop_num)

    cdef void type1(self, Eq func) nogil:
        cdef uint n = rand_i(self.dim)
        cdef uint l_v = 0
        while True:
            func(self, n)
            n = (n + 1) % self.dim
            l_v += 1
            if rand_v() >= self.CR or l_v >= self.dim:
                break

    cdef void type2(self, Eq func) nogil:
        cdef uint n = rand_i(self.dim)
        cdef uint l_v
        for l_v in range(self.dim):
            if rand_v() < self.CR or l_v == self.dim - 1:
                func(self, n)
            n = (n + 1) % self.dim

    cdef void f1(self, uint n) nogil:
        self.tmp[n] = self.best[n] + self.F * (
            self.pool[self.r1, n] - self.pool[self.r2, n])

    cdef void f2(self, uint n) nogil:
        self.tmp[n] = self.pool[self.r1, n] + self.F * (
            self.pool[self.r2, n] - self.pool[self.r3, n])

    cdef void f3(self, uint n) nogil:
        self.tmp[n] = self.tmp[n] + self.F * (self.best[n] - self.tmp[n]
                       + self.pool[self.r1, n] - self.pool[self.r2, n])

    cdef void f4(self, uint n) nogil:
        self.tmp[n] = self.best[n] + self.F * (
            self.pool[self.r1, n] + self.pool[self.r2, n]
            - self.pool[self.r3, n] - self.pool[self.r4, n])

    cdef void f5(self, uint n) nogil:
        self.tmp[n] = self.pool[self.r5, n] + self.F * (
            self.pool[self.r1, n] + self.pool[self.r2, n]
            - self.pool[self.r3, n] - self.pool[self.r4, n])

    cdef inline void recombination(self, int i) nogil:
        """use new vector, recombination the new one member to tmp."""
        self.tmp[:] = self.pool[i, :]
        if self.strategy == 1:
            self.type1(Differential.f1)
        elif self.strategy == 2:
            self.type1(Differential.f2)
        elif self.strategy == 3:
            self.type1(Differential.f3)
        elif self.strategy == 4:
            self.type1(Differential.f4)
        elif self.strategy == 5:
            self.type1(Differential.f5)
        elif self.strategy == 6:
            self.type2(Differential.f1)
        elif self.strategy == 7:
            self.type2(Differential.f2)
        elif self.strategy == 8:
            self.type2(Differential.f3)
        elif self.strategy == 9:
            self.type2(Differential.f4)
        elif self.strategy == 0:
            self.type2(Differential.f5)

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
