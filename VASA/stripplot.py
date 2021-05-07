import seaborn as sns
# from VASA.preprocessing.prep_vasa import VASA

def stripplot(v):

    


sns.set_theme(style="whitegrid")

# Load the dataset
crashes = sns.load_dataset("car_crashes")

# Make the PairGrid
g = sns.PairGrid(plot_data,
                 x_vars=columns, y_vars="letter_code",
                 height=10, aspect=.25, hue="which", palette=dict(hot="red", cold="blue"))
# Draw a dot plot using the stripplot function

cmap = mpl.colors.ListedColormap([("blue" if w == "cold" else "red") for w in plot_data["which"]])

# Use semantically meaningful titles for the columns
# titles = ["Total crashes", "Speeding crashes", "Alcohol crashes", "Not distracted crashes", "No previous crashes"]

g.map(
    sns.stripplot, 
    size=10, 
    orient="h", 
    jitter=False,
    linewidth=1, 
    edgecolor="w",
    alpha=0.6
)
# Use the same x axis limits on all columns and add better labels
g.set(xlim=(-5, 105), xlabel="", ylabel="", title="")

for ax, title, col in zip(g.axes.flat, titles, columns):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    line_area = plot_data[plot_data["which"] == "hot"]
    ax.fill_betweenx(x1=line_area[col], x2=-5, y=line_area["letter_code"], alpha=0.15, color="red")

    line_area = plot_data[plot_data["which"] == "cold"]
    ax.fill_betweenx(x1=line_area[col], x2=-5, y=line_area["letter_code"], alpha=0.15, color="blue")
    ax.margins(y=0.01)

g.set(xlim=(-5, 105), xlabel="", ylabel="", title="")

sns.despine(left=True, bottom=True)

g.fig.subplots_adjust(bottom=0.05)
g.fig.suptitle("% of counties classified as hotspot / coldspot", y=0, va="top")