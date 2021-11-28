import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy.stats import mode
from typing import List

from VASA.vasa import VASA
from VASA.BasePlot import BasePlot


class Scatter(BasePlot):

    def __init__(self, v: VASA, desc=None, figsize=(0, 0), titles: str or List[str] = None):
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
        if titles and len(titles) == len(cols):
            if not isinstance(titles, list):
                titles = [titles]
        else:
            titles = cols
        self.titles = titles

        n_cols = math.ceil(len(cols) / 2)
        n_rows = min(len(cols), 2)

        self.n_cols = n_cols
        self.n_rows = n_rows
        self.figsize = ((n_rows * 4, n_cols * 4)
                        if figsize[0] * figsize[1] <= 0 else figsize)

    def plot(self, highlight: str = "", show: bool = True, add_noise: bool = False, samples = 0, group=False):
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
        fig, axes = plt.subplots(
            self.n_cols,
            self.n_rows,
            figsize=self.figsize,
            sharex=True,
            sharey=True
        )
        self.fig = fig
        self.axes = [axes] if len(self.v.cols) == 1 else axes.flatten()

        count = self.v.reduce("count")
        recent = self.v.reduce('recency')

        df = count.merge(
            recent,
            left_on="fips",
            right_on="fips",
            how="inner",
            suffixes=("_count", "_recency")
        ).reset_index(drop=True)

        if df.shape[0] == 0:
            return

        if highlight != "":
            df = df[[
                self.v.group_summary(c) == highlight for c in df.fips.values
            ]].reset_index(drop=True)

        if samples > 0:
            np.random.seed(self.v.seed)
            to_incl = np.random.choice(np.arange(0, df.shape[0]), size=samples, replace=False)
            df = df.iloc[to_incl, :].reset_index(drop=True)

        for i, ax in enumerate(self.axes):
            col: str = self.v.cols[i]
            title = self.titles[i] if self.titles and len(
                self.titles) >= i + 1 else col

            points = df[[f"{col}_count", f"{col}_recency"]].copy()
            points["count"] = [
                max(c)
                for c in points[f"{col}_count"]
            ]
            points["which"] = [
                (1 if h > c else (np.nan if h == 0 and c == 0 else 0))
                for h, c in points[f"{col}_count"]
            ]
            points = points.rename(
                {f"{col}_recency": "recent"},
                axis="columns"
            )

            points = points[["recent", "count", "which"]].dropna().groupby(
                ["count", "recent"]).agg(np.mean).reset_index()

            if highlight != "" or group:
                self.__draw_lines(highlight, col, ax,
                                  df[[f"{col}_count", "fips"]], f"{col}_count", add_noise, group)

            self.__create_scatter(ax, points, zorder=10)
            self.__axis_format(ax)

            ax.set_title(title)

        self.plotted = True

        if not show:
            super().save_plot(self._desc, '')
            plt.close()

    def __draw_lines(self, highlight, col, ax, df, c, add_noise, group):
        
        # df = df[[self.v.group_summary(f) == highlight for f in df.fips]].reset_index(drop=True)
        to_select = [f in df.fips.values for f in self.v.fips_order]

        lines = np.array(self.v.df[col].tolist())[:, to_select]

        if group:
            group_order = np.array([self.v.group_summary(f) for f in self.v.fips_order])[to_select]
            groups = np.unique(group_order)
            
            output = np.empty((lines.shape[0], len(groups)))

            for i, g in enumerate(groups):
                group_sel = np.where(group_order == g)[0]
                output[:, i] = mode(lines[:, group_sel], axis=1).mode[:, 0]

            lines = output

        lines_rev = lines[::-1, :]
        lines_order = np.argsort(
            lines.shape[0] - np.argmax(lines_rev == 1, axis=0) - 1)
        
        colors = [(1 if a > b else 2) for a, b in df[c]]
        alpha = 1 / len(lines)
        for i in lines_order[::-1]:
            val = colors[i]
            if val == 0:
                continue
            color = "red" if val == 1 else "blue"
            self.__draw_line(ax, lines[:, i], val,
                             color, min(1, alpha), add_noise)

    def __draw_line(self, ax, xs, val, color, alpha, add_noise):
        sig_vals = (xs == val) + 0
        sig_idcs = np.where(sig_vals == 1)[0]

        if len(sig_idcs) == 0:
            return

        start = max(sig_idcs[0] - 1, 0) if len(sig_idcs) > 0 else 0
        stop = sig_idcs[-1] + 1

        # stop line at list sig value
        xs = xs[start:stop]

        ys = np.cumsum(xs == val)

        if add_noise:
            np.random.seed(self.v.seed)
            ys = ys + np.random.normal(0, 0.125, len(ys))

        ax.plot(
            np.arange(start + 1, stop + 1),
            # + np.random.normal(0, 1/16, size=len(xs)),
            ys,
            c=color,
            alpha=alpha
        )

    def __create_scatter(self, ax, df: pd.DataFrame, **kwargs):
        sns.scatterplot(
            x="recent",
            y="count",
            data=df,
            hue="which",
            palette="bwr",
            ax=ax,
            s=30,
            **kwargs
        )

    def __axis_format(self, ax):
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
