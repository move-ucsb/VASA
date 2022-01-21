from __future__ import annotations
from typing import Literal, Tuple, Callable, List

import numpy as np
from numpy.random import permutation
import pandas as pd
import geopandas as gpd
import libpysal as lps
from esda import Moran_Local

from functools import partial
from multiprocessing import cpu_count, Pool
from datetime import datetime as dt

from splot.libpysal import plot_spatial_weights

from .reduce_vasa_df import *
from .missing_value_handling import combine_ma


def check_ran_lisa(func):
    def check(self, *args, **kwargs):

        if not self._ran_lisa:
            raise Exception("VASA object has not ran the lisa method yet")

        return func(self, *args, **kwargs)

    return check


def check_not_ran_lisa(func):
    def check(self, *args, **kwargs):

        if self._ran_lisa:
            raise Exception("VASA object has already ran the .lisa method")

        return func(self, *args, **kwargs)

    return check


def check_ran_grouped(func):
    def check(self, *args, **kwargs):

        if not self._ran_grouped:
            raise Exception("VASA object has not been grouped yet")

        return func(self, *args, **kwargs)

    return check


def check_not_ran_grouped(func):
    def check(self, *args, **kwargs):

        if self._ran_grouped:
            raise Exception("VASA object is already grouped")

        return func(self, *args, **kwargs)

    return check


class VASA:
    """
    A standard data object for VASA plots

    Examples
    --------
    >>> from VASA import VASA
    >>> ...
    >>> Need example for group_summary

    """

    def __init__(
        self,
        df: pd.DataFrame,
        gdf: gpd.GeoDataFrame,
        group_summary: Callable[[str], str] = lambda group: group,
        df_group_col: str = "fips",
        gdf_group_col: str = "fips",
        date_col: str = "date",
        date_format: str = "%Y-%m-%d",
        temp_res: Literal["day", "week", "month", "year"] = "week",
        seed: int | None = None,
    ) -> None:
        """
        Initialize the VASA object with data.

        Parameters
        ----------
        df: str or pd.DataFrame,
            Pandas DataFrame for data in a long format with the date
        gdf: gpd.GeoDataFrame,
            Geopandas DataFrame for shape file
        group_summary: Callable[[str], str]
            Converts values in the group_col to a value for spatial
            grouping level.
            For example, if the group_col has values of GEOID's for
            census blocks and the group_summary level is at the county
            level, then the function should be: lambda x: x[:5], as the
            first 5 letters are the state and county code.
        df_group_col: str = "fips",
            Column name in the Pandas DataFrame with the geometry
            fips code or id
        gdf_group_col: str = "fips",
            Column name in the GeoPandas DataFrame with the geometry
            fips code or id
        date_col: str = "date",
            Name of the column containing the date string in the
            Pandas DataFrame
        date_format: str = "%Y-%m-%d",
            Format of the date to convert, set to an empty string if
            already datetime objects
        temp_res: Literal["day", "week", "month"] = "week"
            Temporal aggregation
        """
        self.df = df.copy()
        self.df_group_col = df_group_col
        self.gdf = gdf
        self.gdf_group_col = gdf_group_col
        self.date_col = date_col
        self.date_format = date_format
        self.temp_res = temp_res
        self.group_summary = group_summary

        self.cols: List[str] = list(
            set(df.columns) - {self.df_group_col, self.date_col}
        )

        # Convert date column to dates
        if date_format != "" and isinstance(self.df[date_col].dtypes, object):
            self.df[self.date_col] = self.df[date_col].apply(
                lambda x: dt.strptime(x, date_format).date()
            )

        # keep track of current state
        self._ran_grouped = False
        self._ran_lisa = False
        self.seed = seed

        self.filter_group().group()

    @staticmethod
    def __get_year_week(date: dt) -> Tuple[int, int]:
        week_num = date.isocalendar()[1]
        year = date.year

        if week_num == 1 and date.month == 12:
            year += 1
        elif week_num == 53 and date.month == 1:
            year -= 1

        # Year comes first so dataframe is sorted chronologically
        return (year, week_num)

    @check_not_ran_grouped
    @check_not_ran_lisa
    def group(self) -> VASA:
        """
        Temporallay aggregates data based on the supplied temp_res

        Returns
        -------
        v: VASA
            Current VASA instance
        """
        # pass in functions other than mean
        agg_dict = dict(
            zip(
                [*self.cols, self.date_col],
                [*np.repeat(np.nanmean, len(self.cols)), "min"],
            )
        )
        if self.temp_res == "day":
            # assign year_month_day
            grouped = (
                self.df.assign(day=self.df[self.date_col].values)
                .groupby(["day", self.df_group_col])
                .agg(agg_dict)
                .reset_index()
                .groupby(["day"])
            )
        elif self.temp_res == "week":
            year_week = [self.__get_year_week(date) for date in self.df[self.date_col]]

            grouped = (
                self.df.assign(year_week=year_week)
                .groupby(["year_week", self.df_group_col])
                .agg(agg_dict)
                .reset_index()
                .groupby(["year_week"])
            )

        elif self.temp_res == "month":
            # assign year_month
            grouped = self.df
        else:
            raise Exception("Incorrect temporal resolution")

        # this is going to have to be based on the map or something
        output = pd.DataFrame()
        for _, group in grouped:
            ordered = pd.merge(
                self.gdf,
                group,
                left_on=self.gdf_group_col,
                right_on=self.df_group_col,
                how="left",
            )
            # check ordered actuall has cols --- fails if not
            output = output.append(
                {
                    "date": ordered.loc[0, self.date_col],
                    **{c: ordered[c] for c in self.cols},
                },
                ignore_index=True,
            )

        self.fips_order: List[str] = ordered[self.gdf_group_col]
        self.df = output
        # return (output, ordered[self.gdf_group_col])

        self._ran_grouped = True
        return self

    # I dont use this anywhere...
    # specify column...
    def get_county(self, fips: int, date="all") -> List[int]:
        i = list(self.fips_order).index(fips)
        return [row[self.cols[0]][i] for _, row in self.df.iterrows()]

    # I dont use this anywhere...
    # df / list idk, specify columns
    def get_week(self, i: int) -> pd.DataFrame:
        return self.df.loc[i, self.cols]

    # not implemented:
    def save_output(self, date, fips, vars):
        return 1

    @check_ran_grouped
    @check_not_ran_lisa
    def __pct_missing(self, func) -> np.array[float]:
        output = []

        for col in self.cols:
            d = self.__get_col_numpy(col)
            n_total = len(d[0])
            n_partial_missing = len(d[:, func(d)][0])
            pct_partial_missing = n_partial_missing / n_total * 100
            output.append(pct_partial_missing)

        return np.array(output)

    def __get_col_numpy(self, col: str) -> np.array:
        return np.array(self.df[col].tolist())

    def pct_partial_missing(self) -> np.array[float]:
        """
        Calculates the percentage of geometries with some (not all) missing
        values over the time period

        Returns
        -------
        np.array[float]
            Array of % of geometries with some (not all) missing values over
            the time period for each data column in the data frame
        """
        return (
            self.__pct_missing(lambda d: np.any(np.isnan(d), axis=0))
            - self.pct_full_missing()
        )

    def pct_full_missing(self) -> np.array[float]:
        """
        Calculates the percentage of geometries with missing values over
        the entire time period

        Returns
        -------
        np.array[float]
            Array of % of geometries with all missing values over
            the time period for each data column in the data frame
        """
        return self.__pct_missing(lambda d: np.all(np.isnan(d), axis=0))

    @check_ran_grouped
    @check_not_ran_lisa
    def drop_missing(self, thresh: float = 0.2) -> VASA:
        """
        Drop geometries based on percentage of missing values over the
        time period

        Parameters
        ----------
        thresh: int
            Drop geometries with proportion of missing values above this value

        Returns
        -------
        v: VASA
            Current VASA instance
        """
        for col in self.cols:
            d = self.__get_col_numpy(col)

            to_keep: np.array[bool] = np.mean(np.isnan(d), axis=0) < thresh

            self.df[col] = d[:, to_keep].tolist()
            new_fips: List[str] = self.fips_order[to_keep]
            self.fips_order = new_fips

            new_fips_df = pd.DataFrame({"new_fips": new_fips}).astype({"new_fips": str})

            self.gdf = (
                self.gdf.astype({self.gdf_group_col: str})
                .merge(
                    new_fips_df,
                    left_on=self.gdf_group_col,
                    right_on="new_fips",
                    how="right",
                )
                .reset_index(drop=True)
            )

        return self

    @check_ran_grouped
    @check_not_ran_lisa
    def impute(self) -> VASA:
        """ "
        Replace missing values with 7-time-period average for each geometry

        Returns
        -------
        v: VASA
            Current VASA instance
        """
        for col in self.cols:
            data = self.__get_col_numpy(col)
            self.df[col] = np.apply_along_axis(combine_ma, 0, data).tolist()

        return self

    @check_ran_grouped
    @check_not_ran_lisa
    def fill_missing(self) -> VASA:
        """
        Replace missing values with the weekly average value for
        other geometries

        Returns
        -------
        v: VASA
            Current VASA instance
        """
        for col in self.cols:
            d = self.__get_col_numpy(col)

            row_means = np.nanmean(d, axis=1)
            inds = np.where(np.isnan(d))
            d[inds] = np.take(row_means, inds[0])

            self.df[col] = d.tolist()

        return self

    @check_not_ran_grouped
    @check_not_ran_lisa
    def filter_group(self) -> VASA:
        """ "
        Filter out geometries in the gdf keeping only geometries that are
        in the same group_summary as geometries with values

        Returns
        -------
        v: VASA
            Current VASA instance
        """
        unique_groups = np.unique(
            [self.group_summary(g) for g in self.df[self.df_group_col]]
        )
        self.gdf = self.gdf[
            [
                (self.group_summary(g) in unique_groups)
                for g in self.gdf[self.gdf_group_col]
            ]
        ]

        return self

    def create_w(
        self,
        k: int = 0,
        band: float = 0,
        type: Literal["queens", "union", "none"] = "none",
        transform: str = "r",
    ) -> lps.weights.W:
        """
        Creates a libpysal weights object based on the GeoDataFrame. This
        function is called automatically when running the lisa method, however
        may be useful to run before to analyse the connections

        Parameters
        ----------
        k: int
            Number of neighbors, k value, for KNN weights
        band: float
            Band threshold for DistanceBand weights
        type: "queens" | "union" | "none"
            Queens weights or if "union" with non-zero k, a union between
            the queens and KNN weights
        transform: str
            Weights standardization transform value. Row standardized
            by default

        Return
        ------
        W: lps.weights.W
            Spatial weights object from above specification
        """
        self.gdf = self.gdf.reset_index(drop=True)

        if k > 0:
            w = lps.weights.KNN.from_dataframe(self.gdf, geom_col="geometry", k=k)

        if band > 0:
            w = lps.weights.DistanceBand.from_dataframe(
                self.gdf, threshold=band, geom_col="geometry"
            )
        elif type == "queens" or type == "union":
            w_queens = lps.weights.Queen.from_dataframe(self.gdf, geom_col="geometry")

            if type == "union" and k > 0:
                w = lps.weights.w_union(w, w_queens)
            else:
                w = w_queens
        elif k <= 0:
            raise Exception("Insufficient arguments for the create_w function.")

        self.W = w
        self.W.transform = transform

        return self.W

    def show_weights_connection(self, **kwargs):
        """
        Shows the weight connection of the passed in geodataframe.

        Call drop_missing() first to see the connection among geometries
        without missing values.

        Parameters
        ----------
        kwargs:
            Weights arguments for the create_w method,
            and plotting arguments for plot_spatial_weights

        Returns
        -------
        tuple[Figure | Unknown, Any | Unknown]
            matplotlib fig and ax
        """
        w_args = dict(
            k=kwargs.pop("k", 0),
            band=kwargs.pop("band", 0),
            type=kwargs.pop("type", "none"),
            transform=kwargs.pop("transform", "r"),
        )
        self.create_w(**w_args)
        return plot_spatial_weights(self.W, self.gdf, **kwargs)

    @check_ran_grouped
    @check_not_ran_lisa
    def lisa(
        self,
        sig: float = 0.5,
        method: Literal["fdr", "bon", "sim"] = "fdr",
        permutations: int = 999,
        filter: bool = True,
        **kwargs
    ) -> VASA:
        """
        Calculates local moran I over the time period.

        Parameters
        ----------
        sig: float = 0.05
            Significance level
            Default is alpha = 0.05
        method: "fdr" | "bon" | "sim" = "fdr"
            Default to using the False Discovery Rate (fdr),
            other options include Bon Ferroni (bon) or just the
            simulated p-values directly from the local moran test (sim)
        permutations: int = 999
            Number of permutations for simulated p-value
        filter: bool = True
            A value of True removes high-low and low-high significant
            spots where a vale of false keeps them.
        kwargs:
            Weights arguments for the create_w method,

        Returns
        -------
        v: VASA
            Current VASA instance
        """
        num_processes = cpu_count()

        self.create_w(**kwargs)

        with Pool(num_processes) as pool:
            for col in self.cols:

                self.df[col] = list(
                    pool.map(
                        partial(
                            self.lisa_func,
                            col=col,
                            W=self.W,
                            sig=sig,
                            permutations=permutations,
                            filter=filter,
                            which=method,
                            seed=self.seed,
                        ),
                        [row for _, row in self.df.iterrows()],
                    )
                )

        self._ran_lisa = True
        return self

    @check_ran_grouped
    @check_ran_lisa
    def reduce(
        self,
        # this could return anything really...
        reduce: (
            Literal["count", "recency", "count_hh", "count_ll", "mode"]
            | Callable[[List[List[int]]], List[int]]
        ),
    ) -> pd.DataFrame:
        """
        Calculates values to describe each geographic unit over the entire time period

        Parameters
        ----------
        reduce: Literal["count", "recency", "count_hh", "count_ll", "mode"]
            | Callable[[List[List[int]]], List[int]]
            The name of the built-in reducing function or a custom one

        Returns
        -------
        pd.DataFrame
            Dataframe reduced to a single row
        """
        copy: pd.DataFrame = self.df[self.cols].copy()

        if reduce == "count":
            reduce = reduce_by_count
        elif reduce == "count_hh":
            reduce = reduce_by_count_hh
        elif reduce == "count_ll":
            reduce = reduce_by_count_ll
        elif reduce == "recency":
            reduce = reduce_by_recency
        elif reduce == "recency_hh":
            reduce = reduce_by_recency_hh
        elif reduce == "recency_ll":
            reduce = reduce_by_recency_ll
        elif reduce == "mode_sig":
            reduce = reduce_by_mode_sig
        elif reduce == "count_combined":
            reduce = reduce_by_count_combined
        elif reduce == "mode":
            reduce = reduce_by_mode

        return copy.agg(reduce).assign(fips=self.fips_order)

    @staticmethod
    def __moran_quadrants(col, W, alpha, permutations, filter, which, seed):
        print("here in __moran_quadrants")
        print(len(col))

        local_moran = Moran_Local(
            col, W, geoda_quads=True, permutations=permutations, seed=seed
        )

        ps = local_moran.p_sim

        if filter:
            qs = VASA.__filter_quadrants(local_moran.q)
        else:
            qs = local_moran.q

        if which == "fdr":
            f = VASA.__false_discovery_rate(ps, alpha)
        elif which == "sim":
            f = [p < alpha for p in ps]
        elif which == "bon":
            f = VASA.__bonferroni(ps, alpha)
        elif which == "all":
            fdr = VASA.__false_discovery_rate(ps, alpha)
            bon = VASA.__bonferroni(ps, alpha)
            sim = [p < alpha for p in ps]

            qs = combine(qs * np.array(sim), qs * np.array(fdr), qs * np.array(bon))
            f = sim
        else:
            raise 'Valid p-value evaluations: "bon", "fdr", or "sim"'

        return list(qs * np.array(f))

    #
    # to remove
    #
    @staticmethod
    def lisa_func(ordered, col, W, sig, permutations, filter, which, seed):
        print("here in lisa_func")
        return VASA.__moran_quadrants(
            ordered[col],
            W,
            sig,
            permutations=permutations,
            filter=filter,
            which=which,
            seed=seed,
        )

    @staticmethod
    def __filter_quadrants(arr):
        return [(a if a < 3 else a) for a in arr]

    @staticmethod
    def __false_discovery_rate(arr, sig):
        df = pd.DataFrame(arr, columns=["p"]).sort_values("p")
        df["i"] = np.arange(1, len(arr) + 1) * sig / len(arr)
        df["sig"] = df["p"] < df["i"]
        return list(df.sort_index()["sig"])

    @staticmethod
    def __bonferroni(arr, sig):
        return list(np.array(arr) < sig / len(arr))


# We don't want to filter


def __combine(sim, fdr, bon):
    return [
        (b + 4 if b != 0 else (f + 2 if f != 0 else s))
        for b, f, s in zip(bon, fdr, sim)
    ]
