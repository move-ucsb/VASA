import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy.stats import mode
from typing import List

from VASA.vasa import VASA
from VASA.BasePlot import BasePlot


# You will need to modify the following methods:
# > plot
# > __draw_lines
# > __draw_line
# > __create_scatter
# > __axis_format


class Scatter(BasePlot):
    def __init__(
        self, v: VASA, desc=None, figsize=(0, 0), titles: str or List[str] = None
    ):
        """
        Create the scatter plot object.

        Parameters
        ----------
        v: VASA
            VASA object where the lisa() method has been called.
        desc: str
            Plot description used when saving to a file
        figsize: (float, float)
            Matplotlib figsize specification. Leave as (0, 0) to default
            to (n_rows * 4, n_cols * 4).
        titles: str | List[str]
            String (optional for a single plot) or list of strings to give as titles
            to the scatter plots. Defaults as the column name
        """
        if not v._ran_lisa:
            raise Exception("VASA object has not ran the lisa method yet")

        super().__init__("scatter")

        self.v: VASA = v
        self.plotted = False
        self.fontsize = 14
        self._desc = desc if desc else "-".join(v.cols)

        cols = v.cols
        if titles:
            if not isinstance(titles, list):
                titles = [titles]
        else:
            titles = cols
        self.titles = titles

        n_cols = math.ceil(len(cols) / 2)
        n_rows = min(len(cols), 2)

        self.n_cols = n_cols
        self.n_rows = n_rows
        self.figsize = (
            (n_rows * 4, n_cols * 4) if figsize[0] * figsize[1] <= 0 else figsize
        )

    def plot(
        self,
        highlight: str = "",
        show: bool = True,
        add_noise: bool = False,
        samples=0,
        group=False,
    ):
        """
        Creates a scatter plot showing hot/cold LISA classifications over
        the time period.

        Parameters
        ----------
        highlight: str
            Geometry group to draw lines for. This value should match
            with a v.group_summary() result. Example: geometries are at
            the county level and the v.group_summary() function returns the
            state code. Then `highlight` should be a two digit number as a
            string specifying the state to highlight the counties of.
        show: bool = True
            Whether to show the plot or save the file.
        add_noise: bool = True
            Add noise to differentiate lines
        """
        #
        # This is the function that is being called -- the main entry point
        # make adjustments as needed.
        #

        # make axis for each value column in the dataframe:
        # if there are multiple columns then we make a scatter plot for each
        fig, axes = plt.subplots(
            self.n_cols, self.n_rows, figsize=self.figsize, sharex=True, sharey=True
        )
        self.fig = fig
        self.axes = [axes] if len(self.v.cols) == 1 else axes.flatten()

        # this is an aggregated version of the dataframe
        df = self.__df_calc(highlight, samples)

        # for each value column, create a scatter plot:
        for i, ax in enumerate(self.axes):
            col: str = self.v.cols[i]  # column name

            # plot title:
            title = self.titles[i] if self.titles and len(self.titles) >= i + 1 else col

            # This is the data frame used for making the points
            #
            # Columns:
            # > count   : y-axis value
            # > recency : x-axis value
            # > which   : proportion [0, 1] where 0 should be a dark blue representing
            #             all cold spots, 1 should be dark red representing hot spots
            #             0.5 should be white/gray/something neutral representing even
            #             amounts of hot spots and cold spots
            points = self.__df_points(df, col)

            if highlight != "" or group:  # draw scatter plot lines:
                self.__draw_lines(
                    col,
                    ax,
                    df[[f"{col}_count", "fips"]],
                    f"{col}_count",
                    add_noise,
                    group,
                )

            # create the plot:
            self.__create_scatter(ax, points, zorder=10)

            # axis labels etc...
            self.__axis_format(ax)
            ax.set_title(title)

        self.plotted = True

        if not show:  # write to file
            super().save_plot(self._desc, "")
            plt.close()

    def __draw_lines(self, col, ax, df, c, add_noise, group):
        # this function figures out which lines to draw and calls the
        # __draw_line function for each one
        # depending on your implementation, you may need to change this function
        # adding additional parameters to this function will be easy,
        # if it doesn't make sense to draw the lines individually, you can
        # remove that function and the for loop

        # If you can, I would try to avoid changing these calculations:
        to_select = [f in df.fips.values for f in self.v.fips_order]

        lines = np.array(self.v.df[col].tolist())[:, to_select]

        if group:
            group_order = np.array(
                [self.v.group_summary(f) for f in self.v.fips_order]
            )[to_select]
            groups = np.unique(group_order)

            output = np.empty((lines.shape[0], len(groups)))

            for i, g in enumerate(groups):
                group_sel = np.where(group_order == g)[0]
                output[:, i] = mode(lines[:, group_sel], axis=1).mode[:, 0]

            lines = output

        lines_rev = lines[::-1, :]

        # we need to order the lines so ones ending earlier are on top
        # each column represents a line
        lines_order = np.argsort(lines.shape[0] - np.argmax(lines_rev == 1, axis=0) - 1)

        # you may have to change the following, depending on the library you use::

        colors = [(1 if a > b else 2) for a, b in df[c]]
        alpha = 1 / len(lines)
        for i in lines_order[::-1]:
            val = colors[i]
            if val == 0:
                continue
            color = "red" if val == 1 else "blue"
            self.__draw_line(ax, lines[:, i], val, color, min(1, alpha), add_noise)

    def __draw_line(self, ax, xs, val, color, alpha, add_noise):
        #
        # NEED TO CHANGE, depending on the library you choose, you may
        # add parameters to this function, add additional functions, or
        # remove entirely
        #

        # this function gives you the list of x values and y values,
        # probably don't need to change
        xs, ys = self.__calc_line(xs, val, add_noise)

        # change this function:
        ax.plot(xs, ys, c=color, alpha=alpha)

    def __create_scatter(self, ax, df: pd.DataFrame, **kwargs):
        #
        # NEED TO CHANGE, depending on the library you choose, you may
        # add parameters to this function, add additional functions, or
        # remove entirely
        #

        # This function creates all of the circles for the scatter plot
        # Will need to change for the library you decide to use

        sns.scatterplot(
            x="recent",
            y="count",
            data=df,
            hue="which",
            palette="bwr",
            ax=ax,
            s=30,
            **kwargs,
        )

    def __axis_format(self, ax):
        #
        # NEED TO CHANGE, depending on the library you choose, you may
        # add parameters to this function, add additional functions, or
        # remove entirely
        #

        # This function just addes labels and makes adjustments to each
        # scatter plot

        _, max_x = ax.get_xlim()
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_x)
        ax.grid(False)

        ax.set_ylabel("Count", fontsize=self.fontsize)
        ax.set_xlabel("Last Week Number", fontsize=self.fontsize)

        import matplotlib.patches as mpatches

        hot_spot = mpatches.Patch(color="red", label="Hotspot")
        cold_spot = mpatches.Patch(color="blue", label="Coldspot")

        ax.legend(handles=[hot_spot, cold_spot])

    #
    # Beyond this point are just calculations that you
    # most likely won't need to adjust:
    #

    def __df_calc(self, highlight, samples):
        # used to calculate position of points across all columns
        # most likely will *not* need to change

        count = self.v.reduce("count")
        recent = self.v.reduce("recency")

        df = count.merge(
            recent,
            left_on="fips",
            right_on="fips",
            how="inner",
            suffixes=("_count", "_recency"),
        ).reset_index(drop=True)

        if df.shape[0] == 0:
            return

        if highlight != "":
            df = df[
                [self.v.group_summary(c) == highlight for c in df.fips.values]
            ].reset_index(drop=True)

        if samples > 0:
            np.random.seed(self.v.seed)
            to_incl = np.random.choice(
                np.arange(0, df.shape[0]), size=samples, replace=False
            )
            df = df.iloc[to_incl, :].reset_index(drop=True)

        return df

    def __df_points(self, df, col):
        points = df[[f"{col}_count", f"{col}_recency"]].copy()
        points["count"] = [max(c) for c in points[f"{col}_count"]]
        points["which"] = [
            (1 if h > c else (np.nan if h == 0 and c == 0 else 0))
            for h, c in points[f"{col}_count"]
        ]
        points = points.rename({f"{col}_recency": "recent"}, axis="columns")

        points = (
            points[["recent", "count", "which"]]
            .dropna()
            .groupby(["count", "recent"])
            .agg(np.mean)
            .reset_index()
        )
        return points

    def __calc_line(self, xs, val, add_noise):
        # given a sequence of [0, 1, 0, 0, 2, 1, 0, ...]
        # and the target categorization: 1 or 2
        # produce the cumulative sum of the count of occurences
        # and the date values changes are on

        sig_vals = (xs == val) + 0
        sig_idcs = np.where(sig_vals == 1)[0]

        if len(sig_idcs) == 0:
            return

        start = max(sig_idcs[0] - 1, 0) if len(sig_idcs) > 0 else 0
        stop = sig_idcs[-1] + 1

        ys = np.cumsum(sig_vals)[start:stop]

        if add_noise:
            np.random.seed(self.v.seed)
            ys = ys + np.random.normal(0, 0.125, len(ys))

        return np.arange(start, stop) + 1, ys
