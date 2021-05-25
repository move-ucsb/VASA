import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from typing import List

from VASA.vasa import VASA
from VASA.BasePlot import BasePlot


class Scatter(BasePlot):

    def __init__(self, v: VASA, cols=None):
        fig, axes = plt.subplots(
            math.ceil(len(v.cols) / 2),
            min(len(v.cols), 2),
            figsize=(8, 8)
        )
        super().__init__("scatter", "scatter_test")

        self.v: VASA = v
        self.fig = fig
        self.axes = [axes] if len(v.cols) == 1 else axes.flatten()
        self.plotted = False

    # plot args for like colors??
    # showLines: bool or List[int] # fips
    def plot(self, titles: str or List[str] = ""):
        count = self.v.reduce("count")
        recent = self.v.reduce('recency')

        df = count.merge(
            recent,
            left_on="fips",
            right_on="fips",
            how="inner",
            suffixes=("_count", "_recency")
        ).reset_index()

        # df["count"] = [
        #     max(c)
        #     for c in df["sheltered_in_place_7days_rolling_avg_count"]
        # ]
        # df["which"] = [
        #     (1 if h > c else 0)
        #     for h, c in df["sheltered_in_place_7days_rolling_avg_count"]
        # ]

        # df = df.rename(
        #     { "sheltered_in_place_7days_rolling_avg_recency": "recent" },
        #     axis="columns"
        # )

        # print(df)

        # points = df.groupby(["count", "recent"]).agg(np.mean)

        print(self.axes)

        for i, ax in enumerate(self.axes):
            col: str = self.v.cols[i]

            # print(col, df.columns)

            points = df[[f"{col}_count", f"{col}_recency"]]
            points["count"] = [
                max(c)
                for c in points[f"{col}_count"]
            ]
            points["which"] = [
                (1 if h > c else 0)
                for h, c in points[f"{col}_count"]
            ]
            points = points.rename(
                {f"{col}_recency": "recent"},
                axis="columns"
            )
            points = points.groupby(["count", "recent"]).agg(np.mean)

            print(points.head())

            self.__create_scatter(ax, points)

        self.plotted = True
        return self.fig

    def __create_scatter(self, ax, df: pd.DataFrame):
        sns.scatterplot(
            x="recent",
            y="count",
            data=df,
            hue="which",
            palette="bwr",
            ax=ax
        )

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
