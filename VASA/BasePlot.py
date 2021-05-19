import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# This is not going to be front facing
# // don't need docs/doc string
# but genearl stuff would be nice

# Probably need to double check the typing here
#
# import matplotlib as mpl
# mpl.axes._subplots.AxesSubplot
# mpl.figure.Figure


class BasePlot:

    def __init__(
        self,
        fig: mpl.figure.Figure,
        folder: str
    ) -> None:
        self.fig = fig
        self.folder = folder

        if not os.path.exists(f"{folder}/"):
            os.makedirs(f"{folder}/")

    def save_plot(
        self, name: str,
        file_ext: str = "png",
        dpi: int = 150,
        subfolder: str = "",
        override: bool = False
    ):
        output = f"{self.folder + ('/' if subfolder else '') + subfolder}/"
        if not os.path.exists(f"{output}/"):
            os.makedirs(f"{output}/")

        file_name = f"{output}/{name}.{file_ext}"

        if not override:
            i = 0
            while os.path.isfile(file_name):
                i += 1
                file_name = f"{output}/{name} ({i}).{file_ext}"

        self.fig.set_facecolor("w")
        self.fig.tight_layout()
        self.fig.savefig(
            file_name,
            bbox_inches='tight',
            dpi=dpi
        )

    def hide_axis(self, ax: plt.Axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
