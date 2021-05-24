from __future__ import annotations
from typing import Literal, Tuple, Callable, List

import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal as lps
from esda import Moran_Local

from functools import partial
# from concurrent import futures
from multiprocessing import cpu_count, Pool
from datetime import datetime as dt

# Commented out for testing...
# from .reduce_vasa_df import *


class VASA:
    """
    A standard data object for VASA plots


    Parameters
    ----------


    Attributes
    ----------


    Examples
    --------
    >>> import libpysal


    """
    def __init__(
        self,
        df: str or pd.DataFrame,
        gdf: gpd.GeoDataFrame,
        df_group_col: str = "fips",
        gdf_group_col: str = "fips",
        date_col: str = "date",
        date_format: str = "%Y-%m-%d",
        temp_res: Literal["day", "week", "month", "year"] = "week"
    ) -> None:
        """
            DOCSTRING
        """
        if isinstance(df, str):
            df = pd.read_csv(df)
        elif not isinstance(df, pd.DataFrame):
            raise Exception("Data not of proper type")

        self.df = df.copy()
        self.df_group_col = df_group_col
        self.gdf = gdf
        self.gdf_group_col = gdf_group_col
        self.date_col = date_col
        self.date_format = date_format
        self.temp_res = temp_res

        self.cols: List[str] = list(
            {*df.columns} - {self.df_group_col, self.date_col}
        )

        # Convert date column to dates
        if isinstance(self.df[self.date_col].dtypes, object):
            self.df[self.date_col] = self.df[self.date_col].apply(
                lambda x: dt.strptime(x, self.date_format).date()
            )
        # NUMPY DATES ??

        # self.group()

    # WE NEED TO CHECK IF THERE IS ONLY ONE GROUP.
    # IF WE ONLY HAVE DATES Jan 1-6, Are these always grouped together?
    def group(self) -> None:
        # pass in functions other than mean
        agg_dict = dict(zip(
            [*self.cols, self.date_col],
            [*np.repeat(np.nanmean, len(self.cols)), "min"]
        ))

        if self.temp_res == "day":
            # assign year_month_day
            grouped = self.df
        elif self.temp_res == "week":
            year_week = [
                get_year_week(date)
                for date in self.df[self.date_col]
            ]

            grouped = self.df \
                .assign(
                    year_week=year_week
                ) \
                .groupby(["year_week", self.df_group_col]) \
                .agg(agg_dict) \
                .reset_index() \
                .groupby(["year_week"])

        elif self.temp_res == "month":
            # assign year_month
            grouped = self.df
        elif self.temp_res == "year":
            # assign year
            grouped = self.df
        else:
            raise Exception("Incorrect temporal resolution")

        # this is going to have to be based on the map or something
        output = pd.DataFrame()
        for _, group in grouped:
            ordered = pd.merge(
                self.gdf, group,
                left_on=self.gdf_group_col, right_on=self.df_group_col,
                how='left'
            )

            output = output.append({
                "date": ordered.loc[0, self.date_col],
                **{c: ordered[c] for c in self.cols}
            }, ignore_index=True)

        self.fips_order = ordered[self.gdf_group_col]
        self.df = output
        # return (output, ordered[self.gdf_group_col])

    # specify column...
    def get_county(self, fips: int, date="all") -> List[int]:
        i = list(self.fips_order).index(fips)
        return [row[self.cols[0]][i] for _, row in self.df.iterrows()]

    # df / list idk, specify columns
    def get_week(self, i: int) -> pd.DataFrame:
        return self.df.loc[i, self.cols[0]]

    def save_output(self, date, fips, vars):
        return 1

    def lisa(self) -> None:
        num_processes = cpu_count()

        W = lps.weights.Queen(self.gdf["geometry"])
        W.transform = 'r'

        with Pool(num_processes) as pool:
            self.df[self.cols[0]] = list(
                pool.map(
                    partial(func, col=self.cols[0],
                            W=W, sig=0.05, which="fdr"),
                    [row for _, row in self.df.iterrows()]
                )
            )

    def reduce(
        self,
        # this could return anything really...
        reduce: (
            Literal["count", "recency", "count_hh", "count_ll", "mode"] |
            Callable[[List[List[int]]], List[int]]
        )
    ) -> pd.DataFrame:
        # Do i really need to copy?
        copy: pd.DataFrame = self.df[self.cols].copy()

        if reduce == "count":
            reduce = reduce_by_count
        elif reduce == "count_hh":
            reduce = reduce_by_count_hh
        elif reduce == "count_ll":
            reduce = reduce_by_count_ll
        elif reduce == "recency":
            reduce = reduce_by_recency
        elif reduce == "mode_sig":
            reduce = reduce_by_mode_sig
        elif reduce == "mode":
            reduce = reduce_by_mode

        return copy \
            .agg(reduce) \
            .assign(fips=self.fips_order)

    # I want to change the name of this
    def agg(self, ag):
        return 1


def func(ordered, col, W, sig, which):
    return moran_quadrants(ordered[col], W, sig, which=which)

# esda fdr is not strictly doing fdr


def false_discovery_rate(arr, sig):
    df = pd.DataFrame(arr, columns=["p"]).sort_values("p")
    df["i"] = np.arange(1, len(arr) + 1) * sig / len(arr)
    df["sig"] = df["p"] < df["i"]
    return list(df.sort_index()["sig"])


def bonferroni(arr, sig):
    return list(np.array(arr) < sig / len(arr))


# We don't want to filter
def filter_quadrants(arr):
    return [(a if a < 3 else a) for a in arr]


# this needs to change if we don't filter
def combine(sim, fdr, bon):
    return [
        (b + 4 if b != 0 else (f + 2 if f != 0 else s))
        for b, f, s in zip(bon, fdr, sim)
    ]

#
#
#
#
#

#
# This function runs the local moran test for us and returns an array of
# the quadrant classification for each county
#
#


#
# STILL GET THE ISSUE WHERE VERYTHING IS A 2
#
def moran_quadrants(col, W, alpha, which):
    local_moran = Moran_Local(col, W, geoda_quads=True,
                              permutations=n_permutations(col))

    ps = local_moran.p_sim
    qs = filter_quadrants(local_moran.q)

    if which == "fdr":
        f = false_discovery_rate(ps, alpha)
    elif which == "sim":
        f = [p < alpha for p in ps]
    elif which == "bon":
        f = bonferroni(ps, alpha)
    elif which == "all":
        fdr = false_discovery_rate(ps, alpha)
        bon = bonferroni(ps, alpha)
        sim = [p < alpha for p in ps]

        qs = combine(
            qs * np.array(sim),
            qs * np.array(fdr),
            qs * np.array(bon)
        )
        f = sim
    else:
        raise 'Valid p-value evaluations: "bon", "fdr", or "sim"'

    return list(qs * np.array(f))


def n_permutations(df):
    return 999  # default value


def get_year_week(date: dt) -> Tuple[int, int]:
    """
    get_year_week [summary]

    Args:
        date (dt): [description]

    Returns:
        Tuple[int, int]: [description]
    """
    week_num = date.isocalendar()[1]
    year = date.year

    if week_num == 1 and date.month == 12:
        year += 1
    elif week_num == 53 and date.month == 1:
        year -= 1

    # Year comes first so dataframe is sorted chronologically
    return (year, week_num)
