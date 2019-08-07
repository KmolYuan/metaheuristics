# -*- coding: utf-8 -*-
# cython: language_level=3

"""The callable class of the validation in algorithm.
The 'verify' module should be loaded when using sub-class.

author: Yuan Chang
copyright: Copyright (C) 2016-2019
license: AGPL
email: pyslvs@gmail.com
"""

from numpy cimport ndarray


cdef enum Limit:
    MAX_GEN
    MIN_FIT
    MAX_TIME


cdef double rand_v(double lower = *, double upper = *) nogil
cdef int rand_i(int upper) nogil


cdef class Chromosome:
    cdef int n
    cdef double f
    cdef ndarray v
    cdef void assign(self, Chromosome obj)


cdef class Verification:
    cdef ndarray[double, ndim=1] get_upper(self)
    cdef ndarray[double, ndim=1] get_lower(self)
    cdef double fitness(self, ndarray[double, ndim=1] v)
    cpdef object result(self, ndarray[double, ndim=1] v)


cdef class AlgorithmBase:

    cdef int max_gen, max_time, gen, rpt
    cdef double min_fit, time_start
    cdef Limit option
    cdef Verification func
    cdef Chromosome last_best
    cdef list fitness_time
    cdef double[:] lb, ub
    cdef object progress_fun, interrupt_fun

    cdef void initialize(self)
    cdef void generation_process(self)
    cdef void report(self)
    cpdef tuple run(self)