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
        super().__init__(fig, "scatter_test")

        self.fig = fig
        self.axes = [axes]
        self.v: VASA = v

    def plot(self):
        return 1
