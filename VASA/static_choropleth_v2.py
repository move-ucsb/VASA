import shapely
import libpysal as lps
from VASA.BasePlot import BasePlot
import math
import geopandas as gpd
import pandas as pd
from matplotlib import colors
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from copy import copy
import json
import altair as alt


class iStackedChoropleth(BasePlot):
    def __init__(self, v):
        """
        Interactive Stacked Choropleth plot showing temporal trends of LISA classifications
        on maps.

        Args:
            v (VASA): VASA object

        Raises:
            Exception: "VASA object has not ran the lisa method yet"
        """
        if not v._ran_lisa:
            raise Exception("VASA object has not ran the lisa method yet")

        self.v = v
        
        # state code dict
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

    #
    #   MAPPING FUNCTIONS
    #

    def plot_count(self, state='', _width=800, _height=600, _titleFontSize = 16):
        """
        Choropleth map showing the total number of times a geometry
        was classified as a hot or cold spots over the time period.

       Args:
            state (str, optional) : state code for state specific plot. Defaults to ''.
            _title (str, optional): plot title. Defaults to ''.
            _width (int, optional): plot width. Defaults to 800.
            _height (int, optional): plot height. Defaults to 600.
            _titleFontSize (int, optional): title font size. Defaults to 16.

        Returns:
            alt.Chart:  Choropleth map showing the total number of times a geometry
                        was classified as a hot or cold spots over the time period.
        """
        _state = state
        __width = _width
        __height =  _height
        __titleFontSize = _titleFontSize
        
        gdf = self.__aggregate_df()
        choro_json = json.loads(gdf.to_json())
        choro_data = alt.Data(values=choro_json['features'])

        return self.__plot_single(choro_data, type='count', state=_state, _width = __width, _height=__height, _titleFontSize=__titleFontSize)

    def plot_recent(self, state='', _width=800, _height=600, _titleFontSize = 16):
        """
        Choropleth map showing the last time a geometry was
        classified as a hot or cold spots over the time period.

        Args:
            state (str, optional) : state code for state specific plot. Defaults to ''.
            _title (str, optional): plot title. Defaults to ''.
            _width (int, optional): plot width. Defaults to 800.
            _height (int, optional): plot height. Defaults to 600.
            _titleFontSize (int, optional): title font size. Defaults to 16.

        Returns:
            alt.Chart:  Choropleth map showing the last time a geometry was
                        classified as a hot or cold spots over the time period.
        """
        _state = state
        __width = _width
        __height =  _height
        __titleFontSize = _titleFontSize
        
        gdf = self.__aggregate_df()
        choro_json = json.loads(gdf.to_json())
        choro_data = alt.Data(values=choro_json['features'])

        return self.__plot_single(choro_data, type='recent', state=_state, _width = __width, _height=__height, _titleFontSize=__titleFontSize)

    def plot_both(self, state='', title='', scheme='equal interval', _width=800, _height=600, _titleFontSize = 16):
        """
        RECO map showing both number of items the geometry was a significant
        hot or cold spot (count) and the last time it was a significant
        value (recency).

        Args:
            state (str, optional) : state code for state specific plot. Defaults to ''.
            _title (str, optional): plot title. Defaults to ''.
            scheme (str, optional): classification scheme for the number of times classified as cold or hot spot, 
                                    choose from 'equal interval', 'quantile'
                                    Defaults to 'equal interval'.
            _width (int, optional): plot width. Defaults to 800.
            _height (int, optional): plot height. Defaults to 600.
            _titleFontSize (int, optional): title font size. Defaults to 16.

        Returns:
            alt.Chart: layered choropleth and scatter map
        """

        # data scource
        gdf = self.__aggregate_df()
        choro_json = json.loads(gdf.to_json())
        choro_data = alt.Data(values=choro_json['features'])
        
        # classification scheme
        if scheme == 'quantile':
            _steps = gdf.total.quantile([0, .25, .5, .75, 1]).to_list() 
        elif scheme == 'equal interval':
             _steps = list(np.round(np.linspace(min(gdf.total), max(gdf.total), 5)))
        else:
            raise Exception("Input classification scheme is invalid")
        
        scheme_param = [_steps, [0,max(gdf.total)], [0, 2*max(gdf.total)]]

        base = self.__base_map(choro_data, state, title, _width, _height)
        hot = self.__plot_hot(choro_data, state, scheme_param)
        cold = self.__plot_cold(choro_data, state, scheme_param)

        # layer combined
        layered = alt.layer(base, cold, hot).resolve_scale(color='independent', strokeOpacity='independent').configure_legend(
            orient='bottom',
            direction='horizontal',
            titleFontSize=8,
            labelFontSize=6,
            padding=10,
            rowPadding=15,
            gradientHorizontalMaxLength=80,
            titleLimit=150
        ).configure_title(
            fontSize=_titleFontSize,
            font='Arial',
            anchor='middle',
            color='black'
        )

        return layered

    #
    #   UTILITY MAPPING FUNCTIONS
    #

    def __aggregate_df(self):
        """
        Generate aggregate dataframe for choropleth plot

        Returns:
            geo dataframe: aggregated geo dataframe with geometric variables and variables of interest
        """

        gdf = self.v.gdf
        gdf = gdf.loc[:, ['STATEFP', 'COUNTYFP', 'GEOID', 'NAME', 'geometry']]
        gdf['centroid'] = gdf.geometry.centroid

        # transfer datum to recommended WGS 84
        # compute centroid for each county
        gdf = gdf.to_crs("EPSG:4326")
        gdf['longitude'] = gdf['centroid'].to_crs("EPSG:4326").x
        gdf['latitude'] = gdf['centroid'].to_crs("EPSG:4326").y
        gdf = gdf.drop(columns='centroid')

        # calculate count and recency
        gdf['hot'] = self.v.reduce("count_hh").iloc[:, 0]
        gdf['cold'] = self.v.reduce("count_ll").iloc[:, 0]
        gdf['total'] = gdf['cold']+gdf['hot']
        gdf['recency'] = self.v.reduce("recency").iloc[:, 0]

        return gdf

    def __base_map(self, data, state='', _title='', _width=800, _height=600):
        """
        Generate base map with county boundary

        Args:
            data (alt.Data): altair data object for generating plot
            state (str, optional) : state code for state specific plot. Defaults to ''.
            _title (str, optional): plot title. Defaults to ''.
            _width (int, optional): plot width. Defaults to 800.
            _height (int, optional): plot height. Defaults to 600.

        Returns:
            alt.Chart: base map
        """

        base = alt.Chart(data).mark_geoshape(
            fill="lightgray",
            stroke='white',
            strokeWidth=0.1
        ).properties(
            width=_width,
            height=_height,
            title=_title
        ).project(
            type='albersUsa'
        )

        if state:
            base = base.transform_filter(
                f"datum.properties.STATEFP == '{self.state_code[state]}'")

        return base

    def __plot_hot(self, data, state='', scheme_params=[]):
        """
        RECO map showing both number of items the geometry was a significant
        hot spot (count) and the last time it was a significant
        value (recency).

        Args:
            data (alt.Data): altair data object for generating plot
            state (str, optional) : state code for state specific plot, refer to self.state_code. Defaults to ''.
            scheme_param (List, optional): classification scheme for the number of times classified as cold or hot spot, 
                                    choose from 'equal interval', 'quantile', 'natural break'. 
                                    Defaults to 'equal interval'.

        Returns:
            alt.Chart: mapped scatter plot with only hot spots
        """
        
        _steps, _domain, _range = scheme_params[0], scheme_params[1], scheme_params[2]

        hot = alt.Chart(data).mark_circle().encode(
            latitude='properties.latitude:Q',
            longitude='properties.longitude:Q',
            stroke=alt.ColorValue("black"),
            strokeWidth=alt.value(1),
            strokeOpacity=alt.condition(
                'datum.properties.recency == 0',
                alt.value(1),
                alt.value(0)),
            color=alt.Color('properties.recency:Q',
                            bin=alt.Bin(step=5),
                            scale=alt.Scale(scheme='reds'),
                            legend=alt.Legend(title='Week Numbers (Hot Spots)')),
            size=alt.Size('properties.total:Q',
                          # classification scheme to be modify here
                          scale=alt.Scale(
                              type='bin-ordinal', bins=_steps, domain=_domain, range=_range),  
                          bin=True,
                          legend=alt.Legend(title='Number of Time Classified as Cold/Hot Spots')),
            tooltip=[alt.Tooltip('properties.NAME:N', title="County"),
                     alt.Tooltip('properties.total:Q', title="Count")]
        ).transform_filter('datum.properties.hot>=datum.properties.cold')

        if state:
            hot = hot.transform_filter(f"datum.properties.STATEFP == '{self.state_code[state]}'")

        return hot

    def __plot_cold(self, data, state='', scheme_params=[]):
        """
        RECO map showing both number of items the geometry was a significant
        cold spot (count) and the last time it was a significant
        value (recency).

        Args:
            data (alt.Data): altair data object for generating plot
            state (str, optional) : state code for state specific plot, refer to self.state_code. Defaults to ''.
            scheme (str, optional): classification scheme for the number of times classified as cold or hot spot, 
                                    choose from 'equal interval', 'quantile', 'natural break'. 
                                    Defaults to 'equal interval'.

        Returns:
            alt.Chart: scatter plot on map with only cold spots
        """
        _steps, _domain, _range = scheme_params[0], scheme_params[1], scheme_params[2]
            
        cold = alt.Chart(data).mark_circle().encode(
            latitude='properties.latitude:Q',
            longitude='properties.longitude:Q',
            stroke=alt.ColorValue("black"),
            strokeWidth=alt.value(1),
            strokeOpacity=alt.condition(
                'datum.properties.recency ==0',  # condition value set to??
                alt.value(1),
                alt.value(0)),
            color=alt.Color('properties.recency:Q',
                            bin=alt.Bin(step=5),
                            scale=alt.Scale(scheme='blues'),
                            legend=alt.Legend(title='Week Numbers (Cold Spots)')),
            size=alt.Size('properties.total:Q',
                          # classification scheme to be modify here
                          scale=alt.Scale(
                              type='bin-ordinal', bins=_steps, domain=_domain, range=_range),  
                          bin=True,
                          legend=alt.Legend(title='Number of Time Classified as Cold/Hot Spots')),
            tooltip=[alt.Tooltip('properties.NAME:N', title="County"),
                     alt.Tooltip('properties.total:Q', title="Count")]
        ).transform_filter('datum.properties.hot<datum.properties.cold')

        if state:
            cold = cold.transform_filter(
                f"datum.properties.STATEFP == '{self.state_code[state]}'")

        return cold

    def __plot_single(self, data, type, state='', _width=800, _height=600, _titleFontSize = 16):
        """
        RECO map showing either number of items the geometry was a significant
        cold spot (count) or the last time it was a significant
        value (recency).

        Args:
            data (alt.Data): altair data object for generating plot
            type (str) : type of choropleth map, "count" or "recent"
            state (str, optional) : state code for state specific plot, refer to self.state_code. Defaults to ''.

        Returns:
            alt.Chart: choropleth map with defined type
        """
        if type == 'count':
            which, _title = 'total', 'Number of Time Classified as Hot or Cold Spot'
        if type == 'recent':
            which, _title = 'recency', 'The Most Recent Week Classified as Hot or Cold Spot'

        display = [alt.Tooltip('properties.NAME:N', title="County"),
                   alt.Tooltip(f'properties.{which}:Q', title="Count")]

        hot = alt.Chart(data).mark_geoshape(
            stroke='white',
            strokeWidth=0.1
        ).encode(
            color=alt.Color(f'properties.{which}:Q',
                            scale=alt.Scale(scheme='reds'),
                            bin=alt.Bin(step=5),
                            legend=alt.Legend(title=_title)),
            tooltip=display
        ).properties(
            width=_width, 
            height=_height,  
            title=_title
        ).transform_filter('datum.properties.hot>=datum.properties.cold')

        cold = alt.Chart(data).mark_geoshape(
            stroke='white',
            strokeWidth=0.1
        ).encode(
            color=alt.Color(f'properties.{which}:Q',
                            scale=alt.Scale(scheme='blues'),
                            bin=alt.Bin(step=5),
                            legend=alt.Legend(title=_title)),
            tooltip=display
        ).properties(
            width=_width, 
            height=_height,  
            title=_title
        ).transform_filter('datum.properties.hot<datum.properties.cold')
        
        if state:
            hot = hot.transform_filter(f"datum.properties.STATEFP == '{self.state_code[state]}'")
            cold = cold.transform_filter(f"datum.properties.STATEFP == '{self.state_code[state]}'")

        both = alt.layer(cold, hot).resolve_scale(color='independent').configure_legend(
            orient='bottom',
            direction='horizontal',
            titleFontSize=8,
            labelFontSize=6,
            padding=10,
            rowPadding=15,
            gradientHorizontalMaxLength=150,
            titleLimit=200
        ).configure_title(
            fontSize=_titleFontSize,
            font='Arial',
            anchor='middle',
            color='black'
        )

        return both

    #
    #   CALCULATIONS:
    #
    def __collapse_count_combined(self):
        self._collapse_count_combined = self.v.reduce("count_combined")

    def __collapse_count(self):
        self._collapse_count_hot = self.v.reduce("count_hh")
        self._collapse_count_cold = self.v.reduce("count_ll")

    def __collapse_recent(self):
        self._collapse_recent_hot = self.v.reduce("recency_hh")
        self._collapse_recent_cold = self.v.reduce("recency_ll")
