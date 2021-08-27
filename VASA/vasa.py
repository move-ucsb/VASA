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

from splot.libpysal import plot_spatial_weights

# Commented out for testing...
from .reduce_vasa_df import *


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
            set(df.columns) - {self.df_group_col, self.date_col}
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

    def pct_partial_missing(self) -> np.array[float]:
        output = []

        for col in self.cols:
            d = np.array(self.df[col].tolist())
            n_total = len(d[0])
            n_partial_missing = len(d[:, np.any(np.isnan(d), axis=0)][0])
            pct_partial_missing = n_partial_missing / n_total * 100
            output.append(pct_partial_missing)

        return np.array(output) - self.pct_full_missing()

    def pct_full_missing(self) -> np.array[float]:
        output = []

        for col in self.cols:
            d = np.array(self.df[col].tolist())
            n_total = len(d[0])
            n_all_missing = len(d[:, np.all(np.isnan(d), axis=0)][0])
            pct_all_missing = n_all_missing / n_total * 100
            output.append(pct_all_missing)

        return np.array(output)

    def drop_missing(self, thresh: int = 0.2):
        for col in self.cols:
            d = np.array(self.df[col].tolist())

            to_keep = np.mean(np.isnan(d), axis=0) < thresh

            self.df[col] = d[:, to_keep].tolist()
            new_fips = self.fips_order[to_keep]
            self.fips_order = new_fips

            new_fips_df = pd.DataFrame({"new_fips": new_fips}) \
                .astype({"new_fips": str})

            self.gdf = self.gdf \
                .astype({self.gdf_group_col: str}) \
                .merge(
                    new_fips_df,
                    left_on=self.gdf_group_col,
                    right_on="new_fips",
                    how="right"
                ) \
                .reset_index(drop=True)

    def impute(self):
        # from sklearn.impute import SimpleImputer

        # imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")

        def moving_average(x):
            return np.convolve(np.nan_to_num(x), np.ones(7), 'same') / 7

        def combine_ma(x):
            to_remove = np.logical_not(np.isnan(x))
            ma = moving_average(x)
            ma[to_remove] = x[to_remove]
            return ma

        for col in self.cols:
            data = np.array(self.df[col].tolist())

            # this is over all the data, could just do the partial missing
            # any_missing = d[:, np.any(np.isnan(d), axis=0)]
            # partial_missing = any_missing[:, np.any(np.logical_not(np.isnan(any_missing)), axis=0)]

            self.df[col] = np.apply_along_axis(combine_ma, 0, data).tolist()

    def __create_w(self, k: int) -> None:
        if k > 0:
            W = lps.weights.KNN.from_dataframe(self.gdf.reset_index(drop=True), "geometry", k=k)
            self.W = W
        else: 
            W = lps.weights.Queen(self.gdf["geometry"])
            W.transform = 'r'
            self.W = W

    def show_weights_connection(self, k: int = 0) -> None:
        self.__create_w(k)
        plot_spatial_weights(self.W, self.gdf)

    def lisa(self, k: int = 0) -> None:
        num_processes = cpu_count()

        self.__create_w(k)

        with Pool(num_processes) as pool:
            for col in self.cols:

                self.df[col] = list(
                    pool.map(
                        partial(func, col=col,
                                W=self.W, sig=0.05, which="fdr"),
                        [row for _, row in self.df.iterrows()]
                    )
                )

    # def lisa(self) -> None:
    #     num_processes = cpu_count()

    #     W = lps.weights.Queen(self.gdf["geometry"])
    #     W.transform = 'r'

    #     def get_order(row):
    #         data = pd.DataFrame({"d": row, "fips": self.fips_order})
    #         l = pd.merge(self.gdf, data, how="left", left_on=self.gdf_group_col, right_on="fips")
    #         print(l["d"])
    #         return l["d"].to_numpy()

    #     with Pool(num_processes) as pool:
    #         for col in self.cols:

    #             self.df[col] = list(
    #                 pool.map(
    #                     partial(func, col=col,
    #                             W=W, sig=0.05, which="fdr"),
    #                     [get_order(row) for _, row in self.df.iterrows()]
    #                 )
    #             )

    def reduce(
        self,
        # this could return anything really...
        reduce: (
            Literal["count", "recency", "count_hh", "count_ll", "mode"] |
            Callable[[List[List[int]]], List[int]]
        )
    ) -> pd.DataFrame:
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
  #  return arr
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

#
#
#

def _filters(n, coverage="usa", excl_non48=True):
    output = True

    if n == 11001: # DC
        output = False
    if excl_non48 and n >= 2000 and n <= 2999: # exclude AK
        output = False
    if excl_non48 and n >= 15001 and n <= 15009: # exclude HI
        output = False
    if n >= 60010: # territories and such
        output = False
    if n == 53055 or n == 25019: # ISLANDS WITH NO NEIGHBORS
        output = False
    if n == 51515: # Bedford County VA, code was changed in 2013
        output = False

    return output

def filter_data(data, fips_col, coverage="usa", excl_non48=True):
    filters = [
        _filters(x[fips_col], coverage, excl_non48) for _, x in data.iterrows()
    ]
    return data.loc[filters]

def filter_map(data, coverage="usa", excl_non48=True):
    # remove ? - I was doing something different for maps...

    filters = [
        _filters(x["fips"], coverage, excl_non48) for _, x in data.iterrows()
    ]
    return data.loc[filters]

