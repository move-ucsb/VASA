# # add state names
# state_names = pd.read_html('https://en.wikipedia.org/wiki/Federal_Information_Processing_Standard_state_code')
# print(len(state_names))

# state_names = state_names[0]

# state_names.columns = ['name', 'letter_abbr', 'num_code', 'stat']

# state_names.drop('stat', axis=1, inplace=True)

# state_names.num_code = state_names.num_code.apply(lambda x: '0' + str(x) if len(str(x))==1 else str(x))

# state_names.head()

# state_names.loc[state_names.num_code=='25']

import matplotlib.pyplot as plt

from VASA.vasa import VASA
from VASA.BasePlot import BasePlot


class Strip(BasePlot):

    def __init__(self, v: VASA, cols=None):
        # fig, axes = plt.subplots(
        #     1,
        #     1,
        #     figsize=(8, 8)
        # )
        super().__init__("scatter", "striptest")

        #self.fig = fig
        #self.axes = [axes]
        self.v: VASA = v

    def plot(self):
        ndf = self.v.reduce("mode")
        ndf['state_num'] = [
            str(f // 1000) for f in ndf.fips
        ]  # .str.slice(start=0, stop=2)
        ndf['state_num'] = [
            ("0" + str(f//1000) if f//1000 < 10 else str(f//1000))
            for f in ndf.fips
        ]
        ndf = pd.merge(
            ndf,
            state_names,
            how='left',
            left_on='state_num',
            right_on='num_code'
        )

        counties_per_state = ndf.groupby(['fips','letter_abbr']) \
            .size() \
            .reset_index() \
            .rename(columns={0:'count'}) \
            .groupby('letter_abbr')['fips'] \
            .size() \
            .reset_index()

        check1 = usa_map \
            .groupby('STATEFP') \
            .size() \
            .reset_index() \
            .set_index("STATEFP") \
            .sort_index()

        check2 = ndf.groupby(['fips','num_code']) \
            .size() \
            .reset_index() \
            .rename(columns={0:'count'}) \
            .groupby('num_code')['fips'] \
            .size() \
            .reset_index() \
            .set_index('num_code') \
            .sort_index()

        #gdf.head()

        check1.columns = ['cnt']

        # print(counties_per_state)

        moder = ndf.loc[ndf[v.cols].sum(axis=1)>0,]
        lst_row = []

        for i in self.v.cols:
            g = moder[['letter_abbr', i]].groupby(['letter_abbr', i]).size().reset_index()
            g['source'] = i
            g.columns = ['letter_abbr', 'val', 'count', 'source']
            lst_row.append(g)

        cc = pd.concat(lst_row) 
        cc = cc.loc[cc.val!=0,]

        print("---- CC.head() ----")
        print(cc.groupby(["source", "val"]).size())

        print(cc.head())
        print(cc.shape)

        ppiv = cc.pivot(index=['letter_abbr', 'val'], columns='source', values='count')
        ppiv = ppiv.reset_index()

        print("--- ppiv.head() ---")
        print(ppiv.head())

        ppiv = ppiv[['letter_abbr', 'val', *self.v.cols]]


        msc = pd.merge(check1, state_names, left_on='STATEFP', right_on='num_code')
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

        print("--ppiv 2222222222 ---")
        print(ppiv.head())

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

        print("HERE")

        # Draw a dot plot using the stripplot function
        g.map(
            sns.stripplot,
            size=10,
            orient="h",
            jitter=False,
            alpha=.65,
            linewidth=1
        )  # marker=r"$\circ$")#, alpha=0.5)

        print("DONE MAPPING")

        # Use the same x axis limits on all columns and add better labels
        g.set(xlim=(-5, 105), xlabel="", ylabel="")

        # Use semantically meaningful titles for the columns
        titles = ['SafeGraph \nmedian distance traveled', "2", "3", "4"]

        for ax, title in zip(g.axes.flat, titles):

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

        # fig.savefig('/strip_plot_v2.jpeg', dpi=150, bbox_inches='tight')

    def save_plot(self, *args, **kwargs):
        #if not self.plotted:
        #    return

        super().save_plot(*args, **kwargs)
