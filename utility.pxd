# -*- coding: utf-8 -*-
# cython: language_level=3

"""The callable class of the validation in algorithm.
The 'utility' module should be loaded when using sub-class.

author: Yuan Chang
copyright: Copyright (C) 2016-2021
license: AGPL
email: pyslvs@gmail.com
"""

from libc.time cimport time_t
from libcpp.list cimport list as clist

ctypedef unsigned int uint

cdef enum Task:
    MAX_GEN
    MIN_FIT
    MAX_TIME
    SLOW_DOWN

cdef packed struct Report:
    uint gen
    double fitness
    double time

cdef double rand_v(double lower = *, double upper = *) nogil
cdef uint rand_i(uint upper) nogil


cdef class ObjFunc:
    cdef uint gen
    cdef double[:] ub
    cdef double[:] lb

    cdef double fitness(self, double[:] v) nogil
    cpdef object result(self, double[:] v)


cdef class Algorithm:
    cdef bint parallel
    cdef uint pop_num, dim, stop_at_i, rpt
    cdef Task stop_at
    cdef double stop_at_f, best_f
    cdef double[:] best, fitness
    cdef double[:, :] pool
    cdef time_t time_start
    cdef clist[Report] fitness_time
    cdef object progress_fun, interrupt_fun
    cdef public ObjFunc func

    # Chromosome
    cdef void new_pop(self)
    cdef double[:] make_tmp(self)
    cdef void assign(self, uint i, uint j) nogil
    cdef void assign_from(self, uint i, double f, double[:] v) nogil
    cdef void set_best(self, uint i) nogil
    cdef void find_best(self) nogil

    cdef void initialize_pop(self) nogil
    cdef void initialize(self) nogil
    cdef void generation_process(self) nogil
    cdef void report(self) nogil
    cpdef list history(self)
    cpdef tuple result(self)
    cpdef object run(self)
