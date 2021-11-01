import shapely
import libpysal as lps
from VASA.BasePlot import BasePlot
import math
import geopandas as gpd
import pandas as pd
from matplotlib import colors
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from matplotlib.colors import ListedColormap


class StackedChoropleth(BasePlot):

    def __init__(self, v, desc="", titles=None, plot_dim=None, figsize=(0, 0)):
        """
        Stacked Choropleth plot showing temporal trends of LISA classifications
        on maps.

        Parameters
        ----------
        v: VASA
            VASA data object with lisa() method completed.
        desc: str
            Description of plot used for file names, defaults to the names
            of the columns on the plot
        titles: str | List[str]
            String for a single plot or list of strings used as titles for the
            choropleth maps
        plot_dim: (int, int)
            Dimension of figures used to change the arrangement of
            multiple maps.
        figsize: (float, float):
            Size of the entire figure, leave as (0, 0) to default
            to (n_rows * 4, n_cols * 4)
        """
        if not v._ran_lisa:
            raise Exception("VASA object has not ran the lisa method yet")

        cols = v.cols

        if titles and len(titles) == len(cols):
            if not isinstance(titles, list):
                titles = [titles]
        else:
            titles = cols

        if plot_dim == None:
            tot = len(cols)
            if tot > 1:
                plot_dim = (math.ceil(tot / 2), 2)
            if tot == 1:
                plot_dim = (1, 1)

        self._gpd_maps = [v.gdf]
        self._cols = cols
        self.v = v
        self._plot_dim = plot_dim
        self._desc = desc if desc else "-".join(cols)
        self._titles = titles

        self._fips_order = v.fips_order

        self.count_subfolder = 'stacked/count/'
        self.recent_subfolder = 'stacked/recent/'
        self.both_subfolder = 'stacked/combined/'

        super().__init__("stacked", titles)

        self._count_labels = ["Low Count", "High Count"]
        self._recent_labels = ["1/1/2020", "12/31/2020"]
        self._count_title = "Number of Weeks"
        self._recent_title = "Recency"

        self._plot_title_size = 14
        self._font_size = 12
        self._tick_size = 8

        n_cols = math.ceil(len(cols) / 2)
        n_rows = min(len(cols), 2)

        self._figsize: tuple[float, float] = (
            (n_rows * 4, n_cols * 4)
            if figsize[0] * figsize[1] <= 0 else figsize
        )

    #
    #   MAPPING FUNCTIONS
    #

    def plot_count(self, show: bool = True):
        """
        Choropleth map showing the total number of times a geometry
        was classified as a hot or cold spots over the time period.

        Parameters
        ----------
        show: boolean = True
            Whether to show the plot (True) or to not show and save the output (False)
        """
        if '_collapse_count_hot' not in locals():
            self.__collapse_count()

        self.__create_choropleth(
            self._collapse_count_hot,
            self._collapse_count_cold,
            typ=self._count_title,
            labels=self._count_labels,  # Start date to End Date
            figsize=self._figsize,
            show=show
        )

    def plot_recent(self, show: bool = True):
        """
        Choropleth map showing the last time a geometry was
        classified as a hot or cold spots over the time period.

        Parameters
        ----------
        show: boolean = True
            Whether to show the plot (True) or to not show and save the output (False)
        """
        if '_collapse_recent_hot' not in locals():
            self.__collapse_recent()

        self.__create_choropleth(
            self._collapse_recent_hot,
            self._collapse_recent_cold,
            typ=self._recent_title,
            labels=self._recent_labels,
            figsize=self._figsize,
            show=show
        )

    def plot_both(self, a: float = 2500, b: float = 275, show: bool = True):
        """
        RECO map showing both number of items the geometry was a significant
        hot or cold spot (count) and the last time it was a significant
        value (recency).

        Parameters
        ----------
        a: float
            Circle marker size intercept parameter. Circle marker size is
            determined by the equation: a + b * (count).
        b: float
            Circle marker size scale parameter. Circle marker size is
            determined by the equation: a + b * (count).
        show: boolean = True
            Whether to show the plot (True) or to not show and save the output (False)
        """
        if '_collapse_count_combined' not in locals():
            self.__collapse_count_combined()

        if '_collapse_recent_hot' not in locals():
            self.__collapse_recent()

        hots = self._collapse_recent_hot
        colds = self._collapse_recent_cold
        fig, axes = plt.subplots(
            self._plot_dim[0],
            self._plot_dim[1],
            figsize=(max(self._figsize[0], 8), self._figsize[1]),
            squeeze=False
        )
        fig.tight_layout()
        axes = axes.ravel()

        plt.subplots_adjust(hspace=0)

        for map_idx, gpd_map in enumerate(self._gpd_maps):

            for j, col in enumerate(self._cols):

                map_copy = gpd_map.copy()

                map_copy["geometry"] = [
                    col.centroid.buffer(a + b * count)
                    for col, count in zip(map_copy['geometry'], self._collapse_count_combined[col])
                ]
                ax = axes[map_idx + j]
                norm = colors.Normalize(
                    vmin=1, vmax=max([*hots[col], *colds[col]]))

                if len(self._titles) > 1:
                    ax.set_title(
                        self._titles[map_idx + j], fontsize=self._plot_title_size)
                super().hide_axis(ax)

                # self.__show_country_outline(ax, gpd_map)
                self.__show_state_outline(ax, gpd_map)
                self.__create_choropleth_map(
                    hots[col], ax, map_copy, cmap=self.__get_pallete("Reds"), norm=norm, edgecolor='white')
                self.__create_choropleth_map(
                    colds[col], ax, map_copy, cmap=self.__get_pallete("Blues"), norm=norm, edgecolor='white')

        if len(self._titles) == 1:
            fig.suptitle(self._titles[0], fontsize=self._plot_title_size)

        self.__create_choropleth_legend_horiz(
            fig, self._recent_title, self._recent_labels)
        self.__create_choropleth_legend_circles(fig, a, b)

        if not show:
            super().save_plot(self._desc, 'combined')
            plt.close()

    def plot_bivar(self, show: bool = True) -> None:
        """
        Bivariate choropleth map showing both number of items the geometry was a significant
        hot or cold spot (count) and the last time it was a significant
        value (recency), binned into 3 groups based on half the week number. The 
        top left quadrant is not available since a geometry cannot be classified
        as a significant value more times than the number of weeks it was a significant
        value.

        Parameters
        ----------
        show: boolean = True
            Whether to show the plot (True) or to not show and save the output (False)
        """
        if '_collapse_count_combined' not in locals():
            self.__collapse_count_combined()

        if '_collapse_recent_hot' not in locals():
            self.__collapse_recent()

        hots = self._collapse_recent_hot
        colds = self._collapse_recent_cold

        fig, axes = plt.subplots(
            self._plot_dim[0],
            self._plot_dim[1],
            figsize=self._figsize,
            squeeze=False
        )
        axes = axes.ravel()
        fig.tight_layout()

        plt.subplots_adjust(hspace=0)

        for map_idx, gpd_map in enumerate(self._gpd_maps):

            for j, col in enumerate(self._cols):

                # Row wise:
                ax = axes[map_idx + j]

                if len(self._titles) > 1:
                    ax.set_title(
                        self._titles[map_idx + j], fontsize=self._plot_title_size)
                super().hide_axis(ax)

                classifications = []

                half = self.v.df.shape[0] / 2
                for hot_class, cold_class, count in zip(hots[col].values, colds[col].values, self._collapse_count_combined[col].values):
                    is_sig = hot_class > 0 or cold_class > 0
                    is_hot = hot_class > cold_class
                    is_cold = not is_hot
                    hot_recent = hot_class > half + 1
                    cold_recent = cold_class > half + 1
                    # top quadrant
                    if not is_sig:
                        classifications.append(0)
                    elif count > half:
                        if is_hot and hot_recent:
                            classifications.append(5)
                        elif is_cold and cold_recent:
                            classifications.append(6)
                        else:
                            classifications.append(0)
                    else:
                        if is_hot:
                            if hot_recent:
                                classifications.append(3)
                            else:
                                classifications.append(1)
                        elif is_cold:
                            if cold_recent:
                                classifications.append(4)
                            else:
                                classifications.append(2)
                        else:
                            classifications.append(0)

                # self.__show_country_outline(ax, gpd_map)
                self.__show_state_outline(ax, gpd_map)
                cmap = ListedColormap(
                    ["white", "#f330d2", "#deebf7", "#fb6a4a",
                        "#6baed6", "#67000d", "#08306b"]
                )
                self.__create_choropleth_map(
                    classifications, ax, gpd_map, cmap=cmap, edgecolor='white'
                )

        if len(self._titles) == 1:
            fig.suptitle(self._titles[0], fontsize=self._plot_title_size)

        self.__create_bivar_legend(
            fig, "Hot Spots", [
                'white', "#67000d", "#f330d2", "#fb6a4a"], "left"
        )
        self.__create_bivar_legend(
            fig, "Cold Spots", [
                'white', "#08306b", "#deebf7", "#6baed6"], "right"
        )

        if not show:
            super().save_plot(self._desc, 'bivar')
            plt.close()

    def separate_hot_cold(self, show: bool = True):
        if '_collapse_count_hot' not in locals():
            self.__collapse_count()

        if '_collapse_recent_hot' not in locals():
            self.__collapse_recent()

        fig, axes = plt.subplots(
            self._plot_dim[0] * self._plot_dim[1],
            2,
            figsize=self._figsize,
            squeeze=False
        )
        axes = axes.ravel()
        fig.tight_layout()
        plt.subplots_adjust(hspace=0)

        hots = self._collapse_count_hot
        colds = self._collapse_count_cold

        # Since I'm restricting maps OR col to be a single item this is really a single loop:
        for map_idx, gpd_map in enumerate(self._gpd_maps):

            for j, col in enumerate(self._cols):
                # Row wise:
                ax_hot = axes[map_idx + j]
                ax_cold = axes[map_idx + j + 1]

                norm = colors.Normalize(vmin=0.5, vmax=max(
                    [*hots[map_idx + j], *colds[map_idx + j]]))
                ax_hot.set_title(
                    self._titles[map_idx + j], fontsize=self._plot_title_size)
                ax_cold.set_title(
                    self._titles[map_idx + j], fontsize=self._plot_title_size)
                ax_hot.set_axis_off()
                ax_cold.set_axis_off()

                # self.__show_country_outline(ax, gpd_map)
                self.__show_state_outline(ax_hot, gpd_map)
                self.__show_state_outline(ax_cold, gpd_map)
                self.__create_choropleth_map(
                    hots[map_idx + j], ax_hot, gpd_map, cmap=self.__get_pallete("Reds"), norm=norm, edgecolor='black')
                self.__create_choropleth_map(
                    colds[map_idx + j], ax_cold, gpd_map, cmap=self.__get_pallete("Blues"), norm=norm, edgecolor='black')

        if not show:
            super().save_plot(self._desc, "separate")
            plt.close()
    #
    #   MAP FEATURES
    #

    def __create_bivar_legend(self, fig, title, colors, position):
        start_x = 0.12
        start_y = -0.07  # 0 for non tight

        if position == "left":
            newa = fig.add_axes([start_x, start_y, 0.35, 0.15])
        elif position == "right":
            newa = fig.add_axes([start_x + 0.5, start_y, 0.35, 0.15])
        fig.patch.set_facecolor('white')

        vals = np.arange(4).reshape(2, 2)
        newa.imshow(vals, cmap=ListedColormap(colors))
        newa.xaxis.set_ticks([0, 1])
        newa.yaxis.set_ticks([0, 1])
        newa.set_xticklabels(["Past", "Recent"], fontsize=self._tick_size)
        newa.set_yticklabels(["High", "Low"], fontsize=self._tick_size)
        newa.set_xlabel("Recency", fontsize=self._font_size)
        newa.set_ylabel("Count", fontsize=self._font_size)
        newa.set_title(title, fontsize=self._font_size)

    def __create_choropleth(self, hots, colds, typ, labels, figsize, show):
        fig, axes = plt.subplots(
            self._plot_dim[0],
            self._plot_dim[1],
            figsize=figsize,
            squeeze=False
        )
        axes = axes.ravel()
        fig.tight_layout()
        plt.subplots_adjust(hspace=0)

        for map_idx, gpd_map in enumerate(self._gpd_maps):

            for j, col in enumerate(self._cols):
                ax = axes[map_idx + j]

                norm = colors.Normalize(
                    vmin=0.5, vmax=max([*hots[col], *colds[col]]))
                ax.set_title(self._titles[map_idx + j],
                             fontsize=self._plot_title_size)
                ax.set_axis_off()

                self.__show_state_outline(ax, gpd_map)
                self.__create_choropleth_map(
                    hots[col], ax, gpd_map, cmap=self.__get_pallete("Reds"), norm=norm, edgecolor='black')
                self.__create_choropleth_map(
                    colds[col], ax, gpd_map, cmap=self.__get_pallete("Blues"), norm=norm, edgecolor='black')

        self.__create_choropleth_legend_horiz(fig, typ, labels)

        if not show:
            super().save_plot(self._desc, typ)
            plt.close()

        return fig, axes

    def __create_choropleth_map(self, data, ax, gpd_map, **kwargs):
        gpd_map \
            .assign(cl=data) \
            .plot(column='cl', k=2, ax=ax, linewidth=0, **kwargs)\


    def __create_choropleth_legend_horiz(self, fig, typ, labels):
        #
        #   THIS NEEDS TO BE LOOKED INTO
        #
        start_x = 0.12
        start_y = -0.07  # 0 for non tight
        fig.patch.set_facecolor('white')

        newa = fig.add_axes([start_x, start_y + 0.04, 0.3, 0.03])
        sm = plt.cm.ScalarMappable(cmap="Reds")
        cb = plt.colorbar(sm, cax=newa, ticks=[], orientation="horizontal")
        cb.ax.tick_params(size=0)
        cb.ax.set_xlabel(typ, fontsize=self._font_size)
        cb.ax.xaxis.set_label_position("top")
        cb.ax.set_ylabel("Hot Spots", rotation=0, labelpad=35,
                         y=0.125, fontsize=self._font_size)

        newa = fig.add_axes([start_x, start_y + 0, 0.3, 0.03])
        sm = plt.cm.ScalarMappable(cmap="Blues")
        cb = plt.colorbar(sm, cax=newa, ticks=[
            0.1, 0.875], orientation="horizontal")
        cb.ax.set_xticklabels(labels, fontsize=self._font_size)
        cb.ax.tick_params(size=0)
        cb.ax.set_ylabel("Cold Spots", rotation=0, labelpad=35,
                         y=0.125, fontsize=self._font_size)

    def __create_choropleth_legend_circles(self, fig, a: float, b: float):
        start_y = -0.06  # 0 for non tight
        start_x = .58  # 0.5 for non tight

        # this needs to be fixed:
        def point_scale(i: float): return (a + b * i) / a

        newa = fig.add_axes([start_x, start_y + 0.07, 0.3, 0.03])
        points = [1, 5, 10, 20, 52]
        point_lgd = [
            plt.scatter([], [], s=point_scale(i), marker='o', color='k')
            for i in points
        ]
        newa.legend(
            point_lgd,
            points,
            frameon=False,
            title=self._count_title,
            ncol=5,
            handlelength=0.1
        )
        newa.set_axis_off()

    #
    #   UTILITY MAPPING FUNCTIONS
    #

    def __show_state_outline(self, ax, gpd_map):
        gpd_map["outline"] = [
            self.v.group_summary(v) for v in self._fips_order
        ]
        gpd_map \
            .dissolve(by="outline") \
            .plot(color="#FFFFFF00", ax=ax, edgecolor='black', linewidth=1)

    def __get_pallete(self, which: str):
        import copy
        palette = copy.copy(plt.get_cmap(which))
        palette.set_under("#FFFFFF00", 0)
        return palette

    #
    #   CALCULATIONS:
    #
    def __collapse_count_combined(self):
        self._collapse_count_combined = self.v.reduce('count_combined')

    def __collapse_count(self):
        self._collapse_count_hot = self.v.reduce("count_hh")
        self._collapse_count_cold = self.v.reduce("count_ll")

    def __collapse_recent(self):
        self._collapse_recent_hot = self.v.reduce("recency_hh")
        self._collapse_recent_cold = self.v.reduce("recency_ll")
