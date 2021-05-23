import matplotlib.pyplot as plt
import os
from typing import List


class BasePlot:
    """The BasePlot class. This is what it does. Test.
    """
    def __init__(
        self,
        folder: str,
        titles: str or List[str],
        file_ext: str = "png",
        dpi: int = 150
    ) -> None:
        self.folder = folder
        self.titles = titles
        self.file_ext = file_ext
        self.dpi = dpi

        if not os.path.exists(f"{folder}/"):
            os.makedirs(f"{folder}/")

    def save_plot(self, name: str, subfolder: str = ""):
        """
        save_plot This is the save_plot function

        Args:
            name (str): [description]
            subfolder (str, optional): [description]. Defaults to "".
        """
        # self._fig.set_facecolor("w")

        output = f"{self.folder + ('/' if subfolder else '') + subfolder}/"
        if not os.path.exists(f"{output}/"):
            os.makedirs(f"{output}/")

        file_name = f"{output}/{name}.{self.file_ext}"

        i = 0
        while os.path.isfile(file_name):
            i += 1
            file_name = f"{output}/{name}{i}.{self.file_ext}"

        plt.savefig(
            file_name,
            bbox_inches='tight',
            dpi=self.dpi
        )

    def hide_axis(self, ax):
        """
        hide_axis This is the hide_axis function

        Args:
            ax ([type]): [description]
        """
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
