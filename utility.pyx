# -*- coding: utf-8 -*-
# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
# cython: initializedcheck=False, nonecheck=False

"""The callable class of the validation in algorithm.
The 'utility' module should be loaded when using sub-class of base classes.

author: Yuan Chang
copyright: Copyright (C) 2016-2021
license: AGPL
email: pyslvs@gmail.com
"""

from numpy import array, zeros, float64 as f64
from libc.math cimport HUGE_VAL
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time, difftime


cdef inline double rand_v(double lower = 0., double upper = 1.) nogil:
    """Random real value between lower <= r <= upper."""
    return lower + <double>rand() / RAND_MAX * (upper - lower)


cdef inline uint rand_i(uint upper) nogil:
    """A random integer between 0 <= r < upper."""
    return rand() % upper


cdef class ObjFunc:
    """Objective function base class.

    It is used to build the objective function for Meta-heuristic Algorithms.
    """

    cdef double fitness(self, double[:] v) nogil:
        with gil:
            raise NotImplementedError

    cpdef object result(self, double[:] v):
        """The result function. Default is the best variable vector `v`."""
        return array(v)


cdef class Algorithm:
    """Algorithm base class.

    It is used to build the Meta-heuristic Algorithms.
    """

    def __cinit__(
        self,
        ObjFunc func not None,
        dict settings not None,
        object progress_fun=None,
        object interrupt_fun=None
    ):
        """Generic settings."""
        srand(time(NULL))
        # object function
        self.func = func
        self.stop_at = 0
        if 'max_gen' in settings:
            self.task = MAX_GEN
            self.stop_at = settings['max_gen']
        elif 'min_fit' in settings:
            self.task = MIN_FIT
            self.stop_at = settings['min_fit']
        elif 'max_time' in settings:
            self.task = MAX_TIME
            self.stop_at = settings['max_time']
        elif 'slow_down' in settings:
            self.task = SLOW_DOWN
            self.stop_at = 1 - settings['slow_down']
        else:
            raise ValueError("please give 'max_gen', 'min_fit' or 'max_time' limit")
        self.pop_num = settings['pop_num']
        self.rpt = settings['report']
        if self.rpt <= 0:
            self.rpt = 10
        self.progress_fun = progress_fun
        self.interrupt_fun = interrupt_fun
        self.dim = len(self.func.ub)
        if self.dim != len(self.func.lb):
            raise ValueError("length of upper and lower bounds must be equal")
        self.fitness = zeros(self.pop_num, dtype=f64)
        self.pool = zeros((self.pop_num, self.dim), dtype=f64)
        self.best_f = HUGE_VAL
        self.best = zeros(self.dim, dtype=f64)
        # setup benchmark
        self.func.gen = 0
        self.time_start = 0
        self.reports = clist[Report]()

    cdef void assign(self, uint i, uint j) nogil:
        """Copy value from j to i."""
        self.fitness[i] = self.fitness[j]
        self.pool[i, :] = self.pool[j, :]

    cdef void assign_from(self, uint i, double f, double[:] v) nogil:
        """Copy value from tmp."""
        self.fitness[i] = f
        self.pool[i, :] = v

    cdef void set_best(self, uint i) nogil:
        """Set as best."""
        self.best_f = self.fitness[i]
        self.best[:] = self.pool[i, :]

    cdef void find_best(self) nogil:
        """Find the best."""
        cdef uint best = 0
        cdef uint i
        for i in range(0, self.pop_num):
            if self.fitness[i] < self.fitness[best]:
                best = i
        if self.fitness[best] < self.best_f:
            self.set_best(best)

    cdef void init_pop(self) nogil:
        """Initialize population."""
        cdef uint best = 0
        cdef uint i, s
        for i in range(self.pop_num):
            for s in range(self.dim):
                self.pool[i, s] = rand_v(self.func.lb[s], self.func.ub[s])
            self.fitness[i] = self.func.fitness(self.pool[i, :])
            if self.fitness[i] < self.fitness[best]:
                best = i
        if self.fitness[best] < self.best_f:
            self.set_best(best)

    cdef void init(self) nogil:
        """Initialize function."""
        pass

    cdef void generation(self) nogil:
        """The process of each generation."""
        with gil:
            raise NotImplementedError

    cdef double check(self, int s, double v) nogil:
        """Check the bounds."""
        if v > self.func.ub[s]:
            return self.func.ub[s]
        elif v < self.func.lb[s]:
            return self.func.lb[s]
        else:
            return v

    cdef inline void report(self) nogil:
        """Report generation, fitness and time."""
        self.reports.push_back(Report(
            self.func.gen,
            self.best_f,
            difftime(time(NULL), self.time_start),
        ))

    cpdef double[:, :] history(self):
        """Return the history of the process.

        The first value is generation (iteration);
        the second value is fitness;
        the third value is time in second.
        """
        return array([
            (report.gen, report.fitness, report.time)
            for report in self.reports
        ], dtype=f64)

    cpdef tuple result(self):
        """Return the best variable vector and its fitness."""
        return array(self.best), self.best_f

    cpdef object run(self):
        """Run and return the result and convergence history.

        The first place of `return` is came from
        calling [`ObjFunc.result()`](#objfuncresult).

        The second place of `return` is a list of generation data,
        which type is `Tuple[int, float, float]]`.
        The first of them is generation,
        the second is fitness, and the last one is time in second.
        """
        # Swap upper and lower bound if reversed
        for i in range(len(self.func.ub)):
            if self.func.ub[i] < self.func.lb[i]:
                self.func.ub[i], self.func.lb[i] = self.func.lb[i], self.func.ub[i]
        # Start
        self.time_start = time(NULL)
        self.init_pop()
        self.init()
        self.report()
        # Iterations
        cdef double diff, best_f
        cdef double last_diff = 0
        while True:
            best_f = self.best_f
            self.func.gen += 1
            self.generation()
            if self.func.gen % self.rpt == 0:
                self.report()
            if self.task == MAX_GEN:
                if self.func.gen >= self.stop_at > 0:
                    break
            elif self.task == MIN_FIT:
                if self.best_f <= self.stop_at:
                    break
            elif self.task == MAX_TIME:
                if difftime(time(NULL), self.time_start) >= self.stop_at > 0:
                    break
            elif self.task == SLOW_DOWN:
                diff = best_f - self.best_f
                if last_diff > 0 and diff / last_diff >= self.stop_at:
                    break
                last_diff = diff
            # progress
            if self.progress_fun is not None:
                self.progress_fun(self.func.gen, f"{self.best_f:.04f}")
            # interrupt
            if self.interrupt_fun is not None and self.interrupt_fun():
                break
        self.report()
        return self.func.result(self.best)
