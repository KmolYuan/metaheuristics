# -*- coding: utf-8 -*-

from typing import TypedDict
from .de import Strategy


class Setting(TypedDict, total=False):
    pop_num: int
    max_gen: int
    min_fit: float
    max_time: float
    slow_down: float
    report: int
    parallel: bool


class DESetting(Setting):
    strategy: Strategy
    F: float
    CR: float


class RGASetting(Setting):
    cross: float
    mutate: float
    win: float
    delta: float


class FASetting(Setting):
    alpha: float
    beta_min: float
    beta0: float
    gamma: float


class TOBLSetting(Setting):
    pass
