#
#
#   GOING TO MOVE TO THE FOLDER
#
#
#



import seaborn as sns
import matplotlib as mpl
import pandas as pd
# from VASA.preprocessing.prep_vasa import VASA

# type = ["point", "area"]
def stripplot(v, withArea=False):
    hots, colds, combined = agg_state_level(v)

    sns.set_theme(style="whitegrid")

    # Make the PairGrid
    g = sns.PairGrid(
        combined,
        x_vars=v.cols,
        y_vars="letter_code",
        height=10, 
        aspect=.25, 
        hue="which", 
        palette=dict(hot="red", cold="blue")
    )

    # cmap = mpl.colors.ListedColormap(
    #     [("blue" if w == "cold" else "red") for w in combined["which"]]
    # )

    g.map(
        sns.stripplot,
        size=10,
        orient="h",
        jitter=False,
        linewidth=1,
        edgecolor="w",
        alpha=0.6
    )

    g.set(xlim=(-5, 105), xlabel="", ylabel="", title="")

    # REPLACE WITH TITLES
    for ax, title, col in zip(g.axes.flat, v.cols, v.cols):
        # Set a different title for each axes
        ax.set(title=title)

        # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

        if withArea:
            ax.fill_betweenx(
                x1=[max(h, 0) for h in hots[col]],
                x2=0,
                y=hots["letter_code"],
                alpha=0.15,
                color="red"
            )

            ax.fill_betweenx(
                x1=[max(c, 0) for c in colds[col]],
                x2=0,
                y=colds["letter_code"],
                alpha=0.15,
                color="blue"
            )
        ax.margins(y=0.01)

    g.set(xlim=(-5, 105), xlabel="", ylabel="", title="")

    sns.despine(left=True, bottom=True)

    g.fig.subplots_adjust(bottom=0.05)
    g.fig.suptitle(
        "% of counties classified as hotspot / coldspot",
        y=0,
        va="top"
    )


def agg_state_level(v):
    state_names = pd.read_csv("../VASA/state_names.csv")
    state_names.astype({ "num_code": int })

    data = v.reduce("mode")
    data["fips"] = data["fips"] // 1000

    hots = data \
        .groupby(["fips"]) \
        .agg(lambda x: reduce_percentage(x, 1)) \
        .merge(state_names, left_on="fips", right_on="num_code")
    hots["which"] = "hot"

    colds = data \
        .groupby(["fips"]) \
        .agg(lambda x: reduce_percentage(x, 2)) \
        .merge(state_names, left_on="fips", right_on="num_code")
    colds["which"] = "cold"

    combined = hots.append(colds)

    return hots, colds, combined


def reduce_percentage(arr, val: int):
    output = np.mean(arr == val) * 100
    for i, o in enumerate(output):
        output[i] = o if o > 0 else -300
    return output
