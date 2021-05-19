import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List

from VASA.preprocessing.vasa import VASA
from VASA.BasePlot import BasePlot


class Scatter(BasePlot):

    def __init__(self, v: VASA, cols=None):
        fig, axes = plt.subplots(
            1,
            1,
            figsize=(8, 8)
        )
        super().__init__(fig, "scatter_test")

        self.v = v
        self.fig = fig
        self.axes = [axes] # axes.flatten()
        self.plotted = False

    # plot args for like colors??
    # showLines: bool or List[int] # fips
    def plot(self, titles: str or List[str]):
        count = self.v.reduce("count")
        recent = self.v.reduce('recency')

        df = count.merge(
            recent,
            left_on="fips",
            right_on="fips",
            how="inner",
            suffixes=("_count", "_recency")
        )
        df["count"] = [
            max(c)
            for c in df["sheltered_in_place_7days_rolling_avg_count"]
        ]
        df["which"] = [
            (1 if h > c else 0)
            for h, c in df["sheltered_in_place_7days_rolling_avg_count"]
        ]

        df = df.rename(
            { "sheltered_in_place_7days_rolling_avg_recency": "recent" },
            axis="columns"
        )

        print(df)

        points = df.groupby(["count", "recent"]).agg(np.mean)

        print(points)
        
        for i, ax in enumerate(self.axes):

            # fig.tight_layout()

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
