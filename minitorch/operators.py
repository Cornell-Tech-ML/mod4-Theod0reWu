"""Collection of the core mathematical operators used throughout the code base."""

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

import math
from typing import Callable, Iterable


def mul(a: float, b: float) -> float:
    """Multiplies a by b"""
    return a * b


def id(a: float) -> float:
    """Returns the input unchanged"""
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers a and b"""
    return a + b


def neg(a: float) -> float:
    """Negates a number"""
    return -1.0 * a


def lt(a: float, b: float) -> float:
    """Returns true if a is less than b"""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Returns true if a is equal to b"""
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Returns the larger of a and b"""
    return b if a < b else a


def is_close(a: float, b: float) -> float:
    """Returns if a and b are less than 1e-2 apart)"""
    return (a - b < 1e-2) and (b - a < 1e-2)


def sigmoid(x: float) -> float:
    """Returns the sigmoid of x"""
    if x >= 0:
        return 1.0 / (1.0 + math.e ** (-x))
    return math.e**x / (1.0 + math.e**x)


def relu(x: float) -> float:
    """Returns the RELU of x"""
    if x > 0:
        return x
    else:
        return 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Returns the log of x"""
    return math.log(x + EPS)


def exp(a: float) -> float:
    """Return a to the eth power"""
    return math.exp(a)


def inv(x: float) -> float:
    """Returns the reciprocal of x"""
    return 1.0 / x


def log_back(a: float, b: float) -> float:
    """Returns the derivative of log times b"""
    return b / (a + EPS)


def inv_back(a: float, b: float) -> float:
    """Computes the derivative of the reciporacal of a times b"""
    return b * -1.0 / a**2


def relu_back(a: float, b: float) -> float:
    """Computes the derivative of the RELU of a times b"""
    if a > 0:
        return b
    else:
        return 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Returns a list after applying func on each element of a given iterable"""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(func(x))
        return ret

    return _map


def zipWith(
    func: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Returns a list after applying func on each element pair of two given iterables"""

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(func(x, y))
        return ret

    return _zipWith


def reduce(
    func: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Applys the reduce operation with func over the iterable"""

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = func(val, l)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Returns a list with each element from l1 negated"""
    return map(neg)(ls)


def addLists(l1: Iterable[float], l2: Iterable[float]) -> Iterable[float]:
    """Returns a list where each element is the sum from elements in the same index of l1 and l2"""
    return zipWith(add)(l1, l2)


def sum(l1: Iterable[float]) -> float:
    """Returns the sum of all the elements in the list using reduce"""
    return reduce(add, 0.0)(l1)


def prod(l1: Iterable[float]) -> float:
    """Returns the product of all the elements in the list using reduce"""
    return reduce(mul, 1.0)(l1)
