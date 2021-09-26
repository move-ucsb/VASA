import matplotlib.pyplot as plt

from VASA.vasa import VASA
from VASA.BasePlot import BasePlot


class Summary(BasePlot):

    def __init__(self, v: VASA, cols=None):
        fig, axes = plt.subplots(
            1,
            1,
            figsize=(8, 8)
        )
        super().__init__("scatter_test", "summary")

        self.fig = fig
        self.axes = [axes]
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

        ndf = ndf[ndf["state_num"] == "06"]["sg_sheltered"]

        dat = pd.concat(
            [self.v.df["date"].reset_index(drop=True), ndf.reset_index(drop=True)],
            axis=1
        )
