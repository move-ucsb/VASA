import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import mode
import math
from typing import List

from VASA.vasa import VASA
from VASA.BasePlot import BasePlot


class Scatter(BasePlot):

    def __init__(self, v: VASA, cols=None):
        super().__init__("scatter", "scatter_test")

        self.v: VASA = v
        self.plotted = False
        self.fontsize = 14

    # plot args for like colors??
    # showLines: bool or List[int] # fips
    def plot(self, highlight: str = "", titles: str or List[str] = ""):
        fig, axes = plt.subplots(
            math.ceil(len(self.v.cols) / 2),
            min(len(self.v.cols), 2),
            figsize=(8, 8),
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
        ).reset_index()

        if df.shape[0] == 0:
            return

        if highlight != "":
            df = df[[
                self.v.group_summary(c) == highlight for c in df.fips.values
            ]].reset_index()

        for i, ax in enumerate(self.axes):
            col: str = self.v.cols[i]

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

            if highlight != "":
                self.__draw_lines(highlight, col, ax,
                                  df[[f"{col}_count", "fips"]], f"{col}_count")

            self.__create_scatter(ax, points, zorder=10)
            self.__axis_format(ax)

        self.plotted = True
        # return self.fig

    def __draw_lines(self, highlight, col, ax, df, c):

        df = df[[self.v.group_summary(f) == highlight for f in df.fips]]

        to_select = [self.v.group_summary(
            f) == highlight for f in self.v.fips_order]
        lines = np.array(self.v.df[col].tolist())[:, to_select]

        # color = mode(lines).mode[0]
        color = [(1 if a > b else 2) for a, b in df[c]]

        # print(df[c])
        # uzip_a, uzip_b = list(zip(*df[c]))
        # uzip = np.array([*uzip_a, *uzip_b])

        # mm = [np.min(uzip[uzip != 0]), np.max(uzip)]
        # mm = [-1000000, np.max(uzip)]

        # mm = [min(np.min(np.array(df[c]))), max(np.max(np.array(df[c])))]
        # print(mm)

        for i, val in enumerate(color):
            if val == 0:
                continue

            # color = "#d3d3d3"

            # count = np.sum(lines[:, i] == val)
            # alpha = 0.05
            alpha = 1

            # if count == mm[0] or count == mm[1]:
            # print(count, mm)
            # if val == 1:
            #     color = "red"
            # else:
            #     color = "blue"
            # alpha = 1
            # print("HERE")
            color = "red" if val == 1 else "blue"
            self.__draw_line(ax, lines[:, i], val, color, alpha)

    def __draw_line(self, ax, xs, val, color, alpha):
        sig_vals = (xs == val) + 0
        sig_idcs = np.where(sig_vals == 1)[0]

        if len(sig_idcs) == 0:
            return

        sig_idcs = sig_idcs[-1] + 1

        # stop line at list sig value
        xs = xs[:sig_idcs]

        ax.plot(
            np.arange(1, len(xs) + 1),
            np.cumsum(xs == val),  # + np.random.normal(0, 1/16, size=len(xs)),
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

        ax.set_ylabel("Count", fontsize=self.fontsize)
        ax.set_xlabel("Last Week Number", fontsize=self.fontsize)

        import matplotlib.patches as mpatches

        hot_spot = mpatches.Patch(color="red", label="Hotspot")
        cold_spot = mpatches.Patch(color="blue", label="Coldspot")

        plt.legend(handles=[hot_spot, cold_spot])

    def save_plot(self, *args, **kwargs):
        if not self.plotted:
            return

        super().save_plot(*args, **kwargs)

    # def create_scatter(ax, df):
    #     #ax.scatter(xs, ys, c="blue", alpha=0.3)
    #     sns.scatterplot(x="recent", y="count", data=df, hue="which", palette="bwr", ax=ax)
    #     # slope = 1 line
    #     # ax.plot(range(0, max(xs) + 1), range(0, max(xs) + 1))

    # def create_scatter(ax, df):
    #     #ax.scatter(xs, ys, c="blue", alpha=0.3)
    #     sns.scatterplot(x="recent", y="count", data=df, hue="which", palette="icefire", ax=ax)
    #     # slope = 1 line
    #     # ax.plot(range(0, max(xs) + 1), range(0, max(xs) + 1))
