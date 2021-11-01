import matplotlib.pyplot as plt

from VASA import VASA
from VASA.BasePlot import BasePlot

import pandas as pd
import seaborn as sns


class Strip(BasePlot):
    """
    Only works for state level
    """

    def __init__(self, v: VASA, desc: str = "", titles=None):
        """
        Stripplot showing state-level LISA classification trends

        Parameters
        ----------
        v: VASA
            VASA data object with the lisa() method completed
        desc: str
            Description used for the image filename, defaults to the
            name of the columns shown on the plot
        titles: str | List[str]
            A string for a single plot or list of strings to title the
            stripplots. Defaults to the VASA column names.
        """
        if not v._ran_lisa:
            raise Exception("VASA object has not ran the lisa method yet")

        super().__init__("strip")

        cols = v.cols
        self.v: VASA = v
        self._desc = desc if desc else "-".join(cols)

        if titles:
            if not isinstance(titles, list):
                titles = [titles]
        else:
            titles = cols

        self.titles = titles

    def plot(self, show: bool = True):
        """
        Show the stripplot

        Parameters
        ----------
        show: bool = True
            Whether to show the plot or to write it to a file
        """
        state_names = pd.read_csv("../data/state_names.csv")
        state_names.num_code = state_names.num_code.apply(
            lambda x: '0' + str(x) if len(str(x)) == 1 else str(x)
        )

        ndf = self.v.reduce("mode")
        ndf['state_num'] = [self.v.group_summary(f) for f in ndf.fips]

        # combine county data with state names
        ndf = pd.merge(
            ndf,
            state_names,
            how='left',
            left_on='state_num',
            right_on='num_code'
        )

        # number of counties per state
        check1 = self.v.gdf \
            .groupby('STATEFP') \
            .size() \
            .reset_index() \
            .set_index("STATEFP") \
            .sort_index()

        check1.columns = ['cnt']

        moder = ndf.loc[ndf[self.v.cols].sum(axis=1) > 0, ]
        lst_row = []

        for i in self.v.cols:
            g = moder[['letter_abbr', i]].groupby(
                ['letter_abbr', i]).size().reset_index()
            g['source'] = i
            g.columns = ['letter_abbr', 'val', 'count', 'source']
            lst_row.append(g)

        cc = pd.concat(lst_row)
        cc = cc.loc[cc.val != 0, ].loc[cc.val <= 2, ]

        ppiv = cc.pivot(index=['letter_abbr', 'val'],
                        columns='source', values='count')
        ppiv = ppiv.reset_index()

        ppiv = ppiv[['letter_abbr', 'val', *self.v.cols]]

        msc = pd.merge(check1, state_names,
                       left_on='STATEFP', right_on='num_code')
        ppiv = pd.merge(ppiv, msc, how='left', on='letter_abbr')

        last_idx = 2 + len(self.v.cols)

        ppiv.iloc[:, 2:last_idx] = ppiv.iloc[:, 2:last_idx].div(
            ppiv.cnt, axis=0
        ) * 100

        ppiv.iloc[:, 2:last_idx] = ppiv.iloc[:, 2:last_idx].apply(
            lambda x: round(x, 1)
        )

        #
        #
        #   PLOT
        #
        #

        sns.set_theme(style="whitegrid")
        #open_circle = mpl.path.Path(vert)

        text_style = dict(
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=12,
            fontfamily='monospace'
        )

        # Make the PairGrid
        g = sns.PairGrid(
            ppiv,
            x_vars=ppiv.columns[2:last_idx],
            y_vars=["letter_abbr"],
            hue='val',
            height=10,
            aspect=.25,
            palette=['red', 'blue']
        )

        # print("HERE")

        g.map(
            sns.stripplot,
            size=10,
            orient="h",
            jitter=False,
            alpha=.65,
            linewidth=1
        )  # marker=r"$\circ$")#, alpha=0.5)

        # Use the same x axis limits on all columns and add better labels
        g.set(xlim=(-5, 105), xlabel="", ylabel="")

        # Use semantically meaningful titles for the columns

        for ax, title in zip(g.axes.flat, self.titles):

            # Set a different title for each axes
            ax.set(title=title)

            # Make the grid horizontal instead of vertical
            ax.xaxis.grid(False)
            ax.yaxis.grid(True)

        sns.despine(left=True, bottom=True)

        fig = plt.gcf()
        self.fig = fig

        fig.text(
            0.5,
            0.02,
            '% of counties classified as hotspot / coldspot',
            ha='center'
        )

        if not show:
            super().save_plot(self._desc, '')
            plt.close()
