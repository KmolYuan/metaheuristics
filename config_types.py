# -*- coding: utf-8 -*-

from typing import TypedDict


class AlgorithmConfig(TypedDict, total=False):
    pop_num: int
    max_gen: int
    min_fit: float
    max_time: float
    slow_down: float
    report: int
    parallel: bool


class DEConfig(AlgorithmConfig):
    strategy: int
    F: float
    CR: float


class GAConfig(AlgorithmConfig):
    cross: float
    mutate: float
    win: float
    delta: float


class FAConfig(AlgorithmConfig):
    alpha: float
    beta_min: float
    beta0: float
    gamma: float


class TOBLConfig(AlgorithmConfig):
    pass
