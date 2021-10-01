import libpysal as lps
from esda import Moran_Local, Moran
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from typing import List
from datetime import datetime as dt

#
#
# Look at this function:
#
#
def group_data(data, cols, date_col='', group_col='fips', by='week', date_format="%Y-%m-%d"):
    """
    Groups data by date to iterate over later. 

    Parameters
    ----------
    data : dataframe
        Must include a date and grouping column along with variables of interest
    cols : str | List[str]
        Column names for the variables of interest
    date_col : str
        Column name containing dates
    group_col : str
        Column name of grouping, i.e. fips code
    by: 'week' | 'day', optional
        How to group data, default: week.

    Returns
    -------
    dataframe
        Dataframe grouped by date
    """
    if not isinstance(cols, list):
        cols = [cols]

    # Convert string dates into date objects
    dates = [dt.strptime(i, date_format).date() for i in data[date_col]]

    if by == 'week':
        # See the function defined below, this becomes an array of (year, week) tuples
        year_week = [get_year_week(date) for date in dates]

        # We need to make a dictionary to give a function to each column to aggregate by
        # aggregate the cols by mean, also get start (min) and end (max) dates for each group
        # in the future we'll pass in the function in case someone wants to use a different function than mean
        # You can use .extend(), but here I use what's called "iterable unpacking" or "splat" to combine the arrays
        # this is what it does: [*[1, 2, 3], 4] is converted to [1, 2, 3, 4]
        agg_dict = dict(zip([*cols, "date_start", "date_end"],
                            [*np.repeat(np.nanmean, len(cols)), "min", "max"]))
        # The result of this is something like: { "sg_sheltered": np.nanmean, .... }
        # Which means that we will aggreaget "sg_sheltered" column by the mean
        # don't worry if this is confusing, it's not very important


        # This is the most important part to understand:
        # 1. We duplicate the date column to date_start and date_end and assign the (year, week) tuples to another column
        # 2. Group by the week and county 
        # 3. Aggregate: compute weekly averages for each county and find the min and max dates in that week
        # 4. Finally group all counties by the week
        grouped = data \
            .assign(
                week_year=year_week,
                date_start=dates,
                date_end=dates
            ) \
            .groupby(["week_year", group_col]) \
            .agg(agg_dict) \
            .reset_index() \
            .groupby(["week_year"])

    elif by == 'day':
        grouped = data \
            .groupby([date_col])
        # In case there are multiple values for each day:
        # grouped = data \
        #    .assign(**{date_col: dates}) \
        #    .groupby([date_col, group_col])[cols].mean() \
        #    .groupby([date_col])

    return grouped

def get_year_week(date):
    """
    Gets the week number of date (starting on monday).
    Parameters
    ----------
    date : date
        Date to convert
    Returns
    -------
    tuple
        A tuple of (year, week_number)
    """
    week_num = date.isocalendar()[1]
    year = date.year

    # The week number can wrap to the next year, need to ensure year does as well.
    # Don't need to worry about wrapping other way
    if week_num == 1 and date.month == 12:
        year += 1

    if week_num == 53 and date.month == 1:
        year -= 1

    # Year comes first so dataframe is sorted chronologically
    return (year, week_num)

#
#
#
#
#

#
# Local moran has issues with missing values and counties that don't have borders,
# We'll have to figure out a better way to deal with this, but for now I'm just
# getting rid of the places that cause me issues:
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

#
#
#
#
#

#
# After doing the local moran test we have to filter out less significant values
# using false_discovery_rate or bonferroni. We also filter out to leave us with only
# hot-hot spots and cold-cold spots. In the future we'll want to include hot-cold and 
# cold-hot spots as well.
#

# esda fdr is not strictly doing fdr
def false_discovery_rate(arr, sig):
    df = pd.DataFrame(arr, columns=["p"]).sort_values("p")
    df["i"] = np.arange(1, len(arr) + 1) * sig / len(arr)
    df["sig"] = df["p"] < df["i"]
    return list(df.sort_index()["sig"])

def bonferroni(arr, sig):
    return list(np.array(arr) < sig / len(arr))

def filter_quadrants(arr):
    return [(a if a < 3 else a) for a in arr]

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

def moran_quadrants(col, W, alpha, which):
    local_moran = Moran_Local(col, W, geoda_quads=True, permutations=n_permutations(col))

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
    return 999 # default value

#
#
#
#
#

#
#
# This is the function we call from the notebook, it iterates over each group from the output
# of our grouping function and runs the local test for each variable.
#
#

def local_moran(grouped, cols, date_col: str, group_col: str, map_data: pd.DataFrame, map_group_col: str, limit=None, sig=0.05, which="fdr"):
    if not isinstance(cols, list):
        cols = [cols]

    output = pd.DataFrame(
        columns=[*cols, 'date']
    )

    W = lps.weights.Queen(map_data["geometry"])
    W.transform = 'r'

    for i, (date, group) in tqdm(enumerate(grouped), total=min(limit, len(grouped)) if limit else len(grouped)):

        if limit and i == limit:
            break 

        if group.isnull().values.any():
            continue

        # it's important we merge on map so grouping order is consistent
        ordered = pd.merge(
            map_data, group, left_on=map_group_col, right_on=group_col, how='left'
        )

        row = {
            'date': [min(group['date_start']), max(group['date_end'])] if ('date_start' in ordered.columns) else min(group[date_col])
        }

        for j, col in enumerate(cols):
            mq = moran_quadrants(ordered[col], W, sig, which=which)
            row[col] = mq
            
        output = output.append(row, ignore_index=True)

    return (output, map_data[map_group_col])
