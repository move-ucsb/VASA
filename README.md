# VASA

The developed Python package, named VASA, will be accessible at https://github.com/move-ucsb/VASA. To demonstrate the applicability of the designed visualizations, the variability in spatiotemporal structure of human mobility patterns during the COVID-19 pandemic in the United States is assessed. VASA offers three novel multivariate visualizations: A stacked recency and consistency map, a line-path scatter plot, and a categorical strip (dot) plot. All three techniques use LISA as the base and utilize local Moran’s I and permuted p-values. The techniques are best suited for analysis of areal data at two levels of analysis: the object-level and the summary-level. The object-level of analysis receives the data at the finest available scale (e.g. county, census blocks, etc.), whereas the summary-level (e.g. state) refers to the less granular spatial units that contain object-level units. The stacked recency and consistency map allows to ascertain the spatiotemporal structure of data at both object- and summary-level. The categorical strip plot allows for comparison of trends at the summary-level. The line-path visualization is better suited for a fine-detail analysis of individual object-level trajectories within a specified summary-level.

The VASA package includes four classes:

- VASA: A class that deals with aggregations and missing values.

And 3 classes for corresponding types of charts:

- StackedChoropleth
- Scatter
- Strip

![](UML/classes.png)

This StackedChoropleth shows the number of times a CBG was classified as a hot or cold spot over the time period.

![](notebooks/stacked/number_of_weeks/distance_traveled_from_home.png)

This StackedChoropleth shows the last week a CBG was classified as a hot or cold spot over the time period.

![](notebooks/stacked/recency/distance_traveled_from_home.png)

This StackedChoropleth shows both the total number of times a county was classified as a hotspot or coldspot and the last week of that classification.

![](notebooks/usa_dot_choropleth.png)

The following choropleth plot shows both the total number of times a CBG was classified as a hotspot or coldspot and the last week of that classification, by binning values and using the 2D color scheme listed.

![](notebooks/stacked/bivar/distance_traveled_from_home.png)

Scatter plots provide an alternative view to the choropleths, allowing the option to highlight geometries grouped together at a higher level.

![](notebooks/california_scatter_plot.png)

In this example of a Stripplot, the percentages of counties that were classified as hotspots or coldspots is shown for each state.

![](notebooks/stripplot.png)
