from matplotlib import collections
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import pandas as pd
import numpy as np
import math
from scipy.stats import mode
from typing import List
import altair as alt
from VASA.vasa import VASA
from VASA.BasePlot import BasePlot


class iScatter(BasePlot):
    def __init__(
        self, v: VASA, titles=""
    ):
        """
        Create the scatter plot object.

        Parameters
        ----------
        v: VASA
            VASA object where the lisa() method has been called.
        desc: str
            Plot description used when saving to a file
        figsize: (float, float)
            Matplotlib figsize specification. Leave as (0, 0) to default
            to (n_rows * 4, n_cols * 4).
        titles: str | List[str]
            String (optional for a single plot) or list of strings to give as titles
            to the scatter plots. Defaults as the column name
        """
        if not v._ran_lisa:
            raise Exception("VASA object has not ran the lisa method yet")

        super().__init__("scatter")

        self.v: VASA = v
        # self.plotted = False
        # self.fontsize = 14
        # self._desc = desc if desc else "-".join(v.cols)

        self.cols = v.cols
        # if titles:
        #     if not isinstance(titles, list):
        #         titles = [titles]
        # else:
        #     titles = cols
        self.titles = titles
        self.state_code = {
            'WA': '53', 'DE': '10', 'DC': '11', 'WI': '55', 'WV': '54', 'HI': '15',
            'FL': '12', 'WY': '56', 'PR': '72', 'NJ': '34', 'NM': '35', 'TX': '48',
            'LA': '22', 'NC': '37', 'ND': '38', 'NE': '31', 'TN': '47', 'NY': '36',
            'PA': '42', 'AK': '02', 'NV': '32', 'NH': '33', 'VA': '51', 'CO': '08',
            'CA': '06', 'AL': '01', 'AR': '05', 'VT': '50', 'IL': '17', 'GA': '13',
            'IN': '18', 'IA': '19', 'MA': '25', 'AZ': '04', 'ID': '16', 'CT': '09',
            'ME': '23', 'MD': '24', 'OK': '40', 'OH': '39', 'UT': '49', 'MO': '29',
            'MN': '27', 'MI': '26', 'RI': '44', 'KS': '20', 'MT': '30', 'MS': '28',
            'SC': '45', 'KY': '21', 'OR': '41', 'SD': '46'
        }
        
        


    def plot(
        self,
        highlight = '',
        visual = '',
        colname = '',
        samples=0
    ):
        """
        Creates a scatter plot showing hot/cold LISA classifications over
        the time period.

        Parameters
        ----------
        highlight: str
            Geometry group to draw lines for. This value should match
            with a v.group_summary() result. Example: geometries are at
            the county level and the v.group_summary() function returns the
            state code. Then `highlight` should be a two digit number as a
            string specifying the state to highlight the counties of.
        visual: str
            type of interactive plot to render: 'selection' or 'mouseover'
        colname: str
            variable name to plot
        """
        highlight = self.state_code.get(highlight, '')
        df = self.__df_calc(highlight, samples)
        
        if not highlight:
            points_df = self.__df_points(df, colname)
            return self.__scatter(points_df, title = self.titles)
        
        
        source_df = self.__aggregate_df(df, colname)
        if visual == 'mouseover':
            return self.__line_scatter_mouseover(source_df, _title = self.titles)
        elif visual == 'selection':
            return self.__line_scatter_selection(source_df, _title = self.titles)
            
    #
    # PLOTTING FUNCTIONS
    #

    def __scatter(self, df, title=''):
        scatter = alt.Chart(df).mark_circle().encode(
            x=alt.X('recent:O', title='Week Number'),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color('which:Q', scale=alt.Scale(
                range=['#0000FF', '#FF0000']), legend=alt.Legend(title='Cold/Hot')),
            tooltip=['which']
        ).configure_axis(grid=False).properties(
            width=450,
            title=title
        )
        
        return scatter


    def __line_scatter_selection(self, df, _title=''):
    
        highlight = alt.selection(type='single', fields=['line_color'], bind='legend')

        selection = alt.selection_multi(
            fields=['line_color'], bind='legend')

        spot_type = alt.condition(selection,
                                alt.Color('line_color:N', 
                                            scale=alt.Scale(range=['blue', 'red']),
                                            legend=alt.Legend(title='Spot Type')),
                                alt.value('lightgray'))

        display = [alt.Tooltip('fips', title="County"), alt.Tooltip('cum_count_x:Q', title="Count")]


        base = alt.Chart(df).encode(
            x=alt.X('week_number', title='Week Number'),
            y=alt.Y('cum_count_x:Q', title='Count'),
            color=spot_type,
            detail='fips',
            tooltip=display
        )


        points = base.mark_circle(size=25).encode(
            opacity=alt.condition(alt.datum.is_end_pt, alt.value(0.8), alt.value(0))
        ).add_selection(
            highlight
        )

        lines = base.mark_trail().encode(
            size=alt.Size('cum_count_x:Q', legend=None, scale=alt.Scale(range=[0,1.5])),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
        ).add_selection(
            selection
        )


        line_scatter = alt.layer(
            points,
            lines
        ).properties(
            width=450,
            title=_title
        ).configure_axis(grid=False)

        return line_scatter


    def __line_scatter_mouseover(self, df, _title=''):
        highlight = alt.selection(type='single', fields=['fips'], on='mouseover', nearest=True)

        selection = alt.selection_multi(
            fields=['fips:N'], on='mouseover', nearest=True)

        spot_type = alt.condition(highlight,
                                alt.Color('line_color:N', 
                                            scale=alt.Scale(range=['blue', 'red']),
                                            legend=alt.Legend(title='Spot Type')),
                                alt.value('lightgray'))

        display = [alt.Tooltip('fips:N', title="County"), alt.Tooltip('cum_count_x:Q', title="Count")]


        base = alt.Chart(df).encode(
            x=alt.X('week_number', title='Week Number'),
            y=alt.Y('cum_count_x:Q', title='Count'),
            color=spot_type,
            detail='fips:N',
            tooltip=display
        )


        points = base.mark_circle(size=25).encode(
            opacity=alt.condition(alt.datum.is_end_pt, alt.value(0.8), alt.value(0))
        ).add_selection(
            highlight
        )

        lines = base.mark_trail().encode(
            size=alt.Size('cum_count_x:Q', legend=None, scale=alt.Scale(range=[0,1.5])),
            opacity=alt.condition(highlight, alt.value(1), alt.value(0.2))
        ).add_selection(
            selection
        )


        line_scatter = alt.layer(
            points,
            lines
        ).properties(
            width=450,
            title=_title
        ).configure_axis(grid=False)

        return line_scatter
    
    
    
    #
    # UTILITY FUNCTIONS
    #


    def __aggregate_df(self, df, colname=''):
        '''def agg_df()

        Parameters
        ----------
        highlight:
        samples:
        colname: 
        '''
        # colnames default by 1st 
        col = self.cols[0]  if not colname else colname
        c = f"{col}_count"

        # calculations
        to_select = [f in df.fips.values for f in self.v.fips_order]
        lines = np.array(self.v.df[col].tolist())[:, to_select]
        fips_order = self.v.fips_order[to_select]

        colors = [(1 if a > b else 2) for a, b in df[c]]

        # initialize dataframe
        source_df = pd.DataFrame({
            'fips': '',
            'week_number': [],
            'cum_count': [],
            'line_color': ''
        })

        for i, fips in enumerate(fips_order):
            val = colors[i]
            if val == 0:
                continue
            color = "Hot" if val == 1 else "Cold"
            
            xs, ys = self.__calc_line(lines[:, i], val, add_noise=True)

            temp_df = pd.DataFrame({
                'fips': fips,
                'week_number': xs,
                'cum_count': ys,
                'line_color': color
            })

            source_df = pd.concat([source_df, temp_df])

        end_count = source_df.groupby('fips').agg(
            {'cum_count': 'max'}).reset_index()
        source_df = source_df.merge(end_count, on='fips', how='left')
        source_df['is_end_pt'] = source_df.cum_count_x == source_df.cum_count_y
        # points = self.__df_points(self.__df_calc(highlight='', samples=0), col)

        return source_df


    def __df_calc(self, highlight, samples):
        '''
        calculate position of points across all columns
        '''

        count = self.v.reduce("count")
        recent = self.v.reduce("recency")

        df = count.merge(
            recent,
            left_on="fips",
            right_on="fips",
            how="inner",
            suffixes=("_count", "_recency"),
        ).reset_index(drop=True)

        if df.shape[0] == 0:
            return

        # need highlight?
        if highlight != "":
            df = df[
                [self.v.group_summary(c) == highlight for c in df.fips.values]
            ].reset_index(drop=True)

        if samples > 0:
            np.random.seed(self.v.seed)
            to_incl = np.random.choice(
                np.arange(0, df.shape[0]), size=samples, replace=False
            )
            df = df.iloc[to_incl, :].reset_index(drop=True)

        return df

    def __calc_line(self, xs, val, add_noise):
        '''
        given a sequence of [0, 1, 0, 0, 2, 1, 0, ...]
        and the target categorization: 1 or 2
        produce the cumulative sum of the count of occurences
        and the date values changes are on
        '''

        sig_vals = (xs == val) + 0
        sig_idcs = np.where(sig_vals == 1)[0]

        if len(sig_idcs) == 0:
            return

        start = max(sig_idcs[0] - 1, 0) if len(sig_idcs) > 0 else 0
        stop = sig_idcs[-1] + 1
        
        # start, end week number
        xs = np.arange(start, stop) + 1 
        
        # cumulative sum of the count of occurences
        ys = np.cumsum(sig_vals)[start:stop]

        if add_noise:
            np.random.seed(self.v.seed)
            ys = ys + np.random.normal(0, 0.125, len(ys))

        return xs, ys
    
    def __df_points(self, df, colname=''):
        col = self.v.cols[0]  if not colname else colname 
        points = df[[f"{col}_count", f"{col}_recency"]].copy()
        points["count"] = [max(c) for c in points[f"{col}_count"]]
        points["which"] = [
            (1 if h > c else (np.nan if h == 0 and c == 0 else 0))
            for h, c in points[f"{col}_count"]
        ]
        points = points.rename({f"{col}_recency": "recent"}, axis="columns")

        points = (
            points[["recent", "count", "which"]]
            .dropna()
            .groupby(["count", "recent"])
            .agg(np.mean)
            .reset_index()
        )
        return points