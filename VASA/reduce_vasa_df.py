# This is not going to be front facing so it's nice if we have
# general comments but odn't need documentation or doc strings

"""
Utility Reducing Functions for VASA Object
-------------------------------------------------------

Three types:
    * Reducing by count:
        Adding up the total number of times the
        county attained some LISA classification.

    * Reducing by recency:
        The last week number of the time a county
        attained some LISA classification.

    * Reducing by mode:
        The most often LISA classification of a county.
"""

from typing import List, Tuple, Callable
from functools import reduce
import numpy as np
from scipy.stats import mode

HC_List = List[Tuple[int, int]]
County_History_List = List[List[int]]
County_list = List[int]


def reduce_by_count(arr: County_History_List) -> HC_List:

    # Start with pairs of 0 for each county
    initial: HC_List = [(0, 0) for _ in range(len(arr[0]))]

    reducer: Callable[[HC_List, List[int]], HC_List] = lambda acc, curr: [
        (a[0] + (c == 1), a[1] + (c == 2)) for a, c in zip(acc, curr)
    ]

    hh_ll: HC_List = reduce(
        reducer,
        arr,
        initial
    )

    return hh_ll


def reduce_by_count_hh(arr: County_History_List) -> County_list:
    return reduce_by_count_equals(arr, 1)


def reduce_by_count_ll(arr: County_History_List) -> County_list:
    return reduce_by_count_equals(arr, 2)


def reduce_by_count_equals(arr: County_History_List, val: int) -> County_list:
    return reduce(
        lambda acc, curr: np.array(acc) + (np.array(curr) == val),
        arr
    )


# ughhh this needs to be made better
# this should return a date for each classification....
def reduce_by_recency(arr: County_History_List) -> County_list:
    return [
        (hh if clas == 1 else (ll if clas == 2 else 0))
        for hh, ll, clas in zip(
            reduce_by_recency_hh(arr),
            reduce_by_recency_ll(arr),
            reduce_by_mode_sig(arr)
        )
    ]


def reduce_by_recency_hh(arr: County_History_List) -> County_list:
    return reduce_by_recency_equals(arr, 1)


def reduce_by_recency_ll(arr: County_History_List) -> County_list:
    return reduce_by_recency_equals(arr, 2)


def reduce_by_recency_equals(
    arr: County_History_List,
    val: int
) -> County_list:
    return [
        max([
            (idx if week[county_idx] == val else 0)
            for idx, week in enumerate(arr)
        ])
        for county_idx in range(len(arr[0]))
    ]


# not really mode b/c prefers sig over non-sig
# change to list for loop thingy
def reduce_by_mode_sig(arr: County_History_List) -> County_list:
    output: County_list = []

    for hh, ll in reduce_by_count(arr):
        region_class = 1 if hh > ll else 2
        output.append(region_class if max(hh, ll) > 0 else 0)

    return output


def reduce_by_mode(arr: County_History_List) -> County_list:
    return [
        mode([
            week[county_idx]
            for week in arr
        ])[0][0]  # argh
        for county_idx in range(len(arr[0]))
    ]
