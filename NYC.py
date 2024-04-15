#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import h3 
import geopy


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv(r"G:\datasets\NYC_Taxi.csv")
df=df.sample(1000)


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.duplicated().any()


# In[7]:


df.isnull().sum()


# In[8]:


df.dtypes


# In[9]:


df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'],format='%Y-%m-%d %H:%M:%S')


# In[10]:


df['dropoff_datetime']=pd.to_datetime(df['dropoff_datetime'],format='%Y-%m-%d %H:%M:%S')


# In[11]:


df.head(1)


# In[12]:


df['store_and_fwd_flag'].value_counts()


# In[13]:


df['vendor_id'].value_counts()


# In[14]:


a1=[]
a2=[]
for index,x in df.iterrows():
    a1.append(h3.geo_to_h3(x['pickup_latitude'],x['pickup_longitude'],resolution=9))
    a2.append(h3.geo_to_h3(x['dropoff_latitude'],x['dropoff_longitude'],resolution=9))    


# In[15]:


df.drop(columns=['id','vendor_id'],inplace=True)


# In[16]:


df.head()


# In[17]:


df['h3_pickup']=a1
df['h3_dropoff']=a2


# In[18]:


df.head(1)


# In[19]:


import folium


# In[20]:


m=folium.Map(location=[df['pickup_latitude'].mean(),df['pickup_longitude'].mean()],zoom_start=7)


# In[21]:


for index,x in df.iterrows():
    vertices=h3.h3_to_geo_boundary(x['h3_pickup'])
    folium.Polygon(vertices,color='red',fill=True,fill_opacity=0.3).add_to(m)


# In[22]:


m


# In[23]:


m2=folium.Map(location=[df['pickup_latitude'].mean(),df['pickup_longitude'].mean()],zoom_start=7)


# In[24]:


for index,x in df.iterrows():
    vertices=h3.h3_to_geo_boundary(x['h3_dropoff'])
    folium.Polygon(vertices,color='blue',fill=True,fill_opacity=0.3).add_to(m2)


# In[25]:


m2


# In[26]:


from folium.plugins import HeatMap


# In[27]:


m3=folium.Map(location=[df['pickup_latitude'].mean(),df['pickup_longitude'].mean()],zoom_start=7)


# In[28]:


HeatMap(df[['pickup_latitude','pickup_longitude']]).add_to(m3)


# In[29]:


m3


# In[30]:


m4=folium.Map(location=[df['pickup_latitude'].mean(),df['pickup_longitude'].mean()],zoom_start=7)


# In[31]:


HeatMap(df[['dropoff_latitude','dropoff_longitude']]).add_to(m4)


# In[32]:


m4


# In[33]:


from folium.plugins import FastMarkerCluster


# In[34]:


m5=folium.Map(location=[df['pickup_latitude'].mean(),df['pickup_longitude'].mean()],zoom_start=7)


# In[35]:


FastMarkerCluster(df[['pickup_latitude','pickup_longitude']]).add_to(m5)


# In[36]:


m5


# In[37]:


m6=folium.Map(location=[df['pickup_latitude'].mean(),df['pickup_longitude'].mean()],zoom_start=7)


# In[38]:


FastMarkerCluster(df[['dropoff_latitude','dropoff_longitude']]).add_to(m6)


# In[39]:


m6


# In[40]:


from math import sin,cos,sqrt,atan2,radians
def h3_distance(h1,h2):
    lat1,long1=h3.h3_to_geo(h1)
    lat2,long2=h3.h3_to_geo(h2)
    lat1, lat2, long1, long2=map(radians, [lat1, lat2, long1, long2])
    R=6371
    dlat=lat2-lat1
    dlong=long2-long1
    a=sin(dlat/2)**2+cos(lat1)*cos(lat2)*sin(dlong/2)**2
    c=2*atan2(sqrt(a),sqrt(1-a))
    dist=R*c
    return dist


# In[41]:


d=[]
for index,x in df.iterrows():
    d.append(h3_distance(x['h3_pickup'],x['h3_dropoff']))


# In[42]:


df['distance']=d


# In[43]:


df


# In[44]:


df['trip_duration'].describe()


# In[45]:


df=df[df['trip_duration']<20000]


# In[46]:


sns.scatterplot(data=df,x='trip_duration',y='distance')


# In[47]:


import ipywidgets as widgets
from IPython.display import display


# In[48]:


def create_path(sample):
    start=df.iloc[sample]['h3_pickup']
    end=df.iloc[sample]['h3_dropoff']
    vert1=h3.h3_to_geo_boundary(start)
    vert2=h3.h3_to_geo_boundary(end)
    m=folium.Map(location=[df.iloc[sample]['pickup_latitude'],df.iloc[sample]['pickup_longitude']],zoom_start=10)
    folium.Marker(location=[df.iloc[sample]['pickup_latitude'],df.iloc[sample]['pickup_longitude']],popup='Start Point').add_to(m)
    folium.Marker(location=[df.iloc[sample]['dropoff_latitude'],df.iloc[sample]['dropoff_longitude']],popup='End Point').add_to(m)
    folium.PolyLine(locations=[(df.iloc[sample]['pickup_latitude'],df.iloc[sample]['pickup_longitude']),(df.iloc[sample]['dropoff_latitude'],df.iloc[sample]['dropoff_longitude'])],popup=df.iloc[sample]['distance']).add_to(m)
    #folium.Polygon(vert1,color='red',fill=True,fill_opacity=0.3).add_to(m)
    #folium.Polygon(vert2,color='red',fill=True,fill_opacity=0.3).add_to(m)
    
    polyline_points = [(df.iloc[sample]['pickup_latitude'], df.iloc[sample]['pickup_longitude']),(df.iloc[sample]['dropoff_latitude'], df.iloc[sample]['dropoff_longitude'])]
    hexagons=set()
    for point in polyline_points:
        h3_index = h3.geo_to_h3(point[0], point[1], resolution=12)
        hexagons_at_point = h3.hex_range(h3_index, 3)
        hexagons.update(hexagons_at_point)
    for hexagon in hexagons:
        vertices = h3.h3_to_geo_boundary(hexagon)
        folium.Polygon(locations=vertices, color='green', fill=True, fill_opacity=0.3).add_to(m)
    return m


# In[49]:


create_path(5)


# In[50]:


x=list(np.arange(len(df)))


# In[51]:


def widget(change):
    if change['type'] == 'change' and change['name'] == 'value':
        selected_sample = change['new']
        m = create_path(selected_sample)
        display(m)


# In[52]:


ddown=widgets.Dropdown(options=x,description='Sample No.')
result=widgets.Label()
ddown.observe(widget,names='value')


# In[53]:


display(ddown)


# In[54]:


df.plot.scatter(x='pickup_latitude',y='pickup_longitude',figsize=(5,5),alpha=0.1)


# In[55]:


df.plot.scatter(x='dropoff_latitude',y='dropoff_longitude',figsize=(5,5),alpha=0.1)


# In[56]:


vertices_pickup=[]
vertices_dropoff=[]
for index,x in df.iterrows():
    vertices_pickup.append(h3.h3_to_geo_boundary(x['h3_pickup']))
    vertices_dropoff.append(h3.h3_to_geo_boundary(x['h3_dropoff']))


# In[57]:


df['geometry_pickup']=vertices_pickup
df['geometry_dropoff']=vertices_dropoff


# In[58]:


df.head()


# In[59]:


import geopandas as gpd
from shapely.geometry import Polygon


# In[60]:


df['geometry_pickup']=df['geometry_pickup'].apply(lambda x: Polygon(x))
df['geometry_dropoff']=df['geometry_dropoff'].apply(lambda x: Polygon(x))


# Passenger Count Based On Pickup

# In[61]:


df.head()


# In[62]:


df.dtypes


# In[63]:


gdfp=df[['pickup_latitude','pickup_longitude','h3_pickup','geometry_pickup','passenger_count']]
gdfd=df[['dropoff_latitude','dropoff_longitude','h3_dropoff','geometry_dropoff','passenger_count']]


# In[64]:


gdfp=gpd.GeoDataFrame(gdfp,geometry='geometry_pickup')


# In[65]:


gdfd=gpd.GeoDataFrame(gdfd,geometry='geometry_dropoff')


# In[66]:


mz=folium.Map(location=[df['pickup_latitude'].mean(),df['pickup_longitude'].mean()],zoom_start=10)


# In[67]:


d=gdfp.groupby('h3_pickup')['passenger_count'].sum()


# In[68]:


gdfp['Count_Sum']=gdfp['h3_pickup'].apply(lambda x: d[x])


# In[69]:


gdfp.head()


# In[70]:


HeatMap(gdfp[['pickup_latitude','pickup_longitude','Count_Sum']]).add_to(mz)


# In[71]:


mz


# In[72]:


gdfp.iloc[0]['geometry_pickup'].__geo_interface__


# In[73]:


gdfp.dtypes


# In[74]:


for index,x in gdfp.iterrows():
    h=h3.geo_to_h3(float(x['pickup_latitude']),float(x['pickup_longitude']),8)
    vertices=h3.h3_to_geo_boundary(h)
    folium.Polygon(vertices,color='red',fill=False).add_to(mz)


# In[75]:


mz


# Basic Functions 

# In[76]:


h='892a1008b8fffff'


# In[77]:


h3.h3_get_resolution(h)


# In[78]:


h3.h3_get_base_cell(h)


# In[79]:


h3.string_to_h3(h)


# In[80]:


h3.h3_to_string(617733122616721407)


# In[81]:


h3.h3_is_valid(h)


# In[82]:


h3.h3_is_pentagon(h)


# Grid Traversal Functions

# In[83]:


h3.k_ring(h,1)


# In[84]:


h3.h3_to_parent(h)


# In[85]:


h3.h3_to_children(h)


# In[86]:


h3.h3_get_resolution(h) - h3.h3_get_resolution(list(h3.h3_to_children(h))[0])


# In[87]:


h3.edge_length(5)


# Formula for Area = area = 3 * (3 ** 0.5) / 2 * side_length ** 2

# In[88]:


import osmnx as ox


# Finding Routes from one hex to another

# In[89]:


df.head()


# In[90]:


def route_plot(lat1,long1,lat2,long2):
    G=ox.graph_from_place('New York, USA',network_type='drive')
    G=ox.routing.add_edge_speeds(G)
    G=ox.routing.add_edge_travel_times(G)
    origin=ox.distance.nearest_nodes(G,X=long1,Y=lat1)
    destination=ox.distance.nearest_nodes(G,X=long2,Y=lat2)
    route=ox.shortest_path(G,origin,destination,weight='travel_time')
    summary=ox.routing.route_to_gdf(G,route)
    dist=summary['length'].sum()/1000
    m=folium.Map(location=[df['pickup_latitude'].mean(),df['pickup_longitude'].mean()],zoom_start=10)
    folium.Marker(location=[lat1,long1],popup='Start').add_to(m)
    folium.Marker(location=[lat2,long2],popup='End').add_to(m)
    hex1=h3.geo_to_h3(lat1,long1,12)
    hex2=h3.geo_to_h3(lat2,long2,12)
    vert1=h3.h3_to_geo_boundary(hex1)
    vert2=h3.h3_to_geo_boundary(hex2)
    folium.Polygon(vert1,fill=True,color='green',fill_opacity=0.3).add_to(m)
    folium.Polygon(vert2,fill=True,color='green',fill_opacity=0.3).add_to(m)
    for index,row in summary.iterrows():
        coordinates=row['geometry'].coords
        co_list=[[c[1],c[0]] for c in coordinates]
        folium.PolyLine(co_list,popup=dist).add_to(m)
    return m


# In[91]:


dist_map=route_plot(40.736713,-73.997238,40.736492,-73.984741)


# In[92]:


dist_map


# In[93]:


def widget_route(change):
    if change['type'] == 'change' and change['name'] == 'value':
        selected_sample = change['new']
        m = route_plot_first(selected_sample)
        display(m)


# In[94]:


def route_plot_first(sample):
    lat1=df.iloc[sample]['pickup_latitude']
    lat2=df.iloc[sample]['dropoff_latitude']
    long1=df.iloc[sample]['pickup_longitude']
    long2=df.iloc[sample]['dropoff_longitude']
    m=route_plot(lat1,long1,lat2,long2)
    return m


# In[95]:


y=list(np.arange(len(df)))
ddown2=widgets.Dropdown(options=y,description='Sample No.')
result=widgets.Label()
ddown2.observe(widget_route,names='value')


# In[96]:


display(ddown2)


# Finding if the cabs are booked more from places which are far from bus stands or is it non related

# In[97]:


place='New York, USA'
tags={'highway':'bus_stop'}
gdf=ox.features_from_place(place,tags)


# In[98]:


gdf


# In[99]:


gdf.plot(figsize=(100,100))


# In[104]:


m=folium.Map(location=[df['pickup_latitude'].mean(),df['pickup_longitude'].mean()],zoom_start=10)
for index,row in gdf.iterrows():
    geometry=row['geometry']
    lat,long=geometry.y,geometry.x
    h3_index=h3.geo_to_h3(lat,long,14)
    vert=h3.h3_to_geo_boundary(h3_index)
    folium.Polygon(vert,fill=True,color='red',fill_opacity=0.1).add_to(m)


# In[105]:


m


# In[106]:


HeatMap(gdfp[['pickup_latitude','pickup_longitude','Count_Sum']]).add_to(m)


# In[107]:


m


# In[117]:


d=df[['pickup_datetime','pickup_latitude','pickup_longitude','passenger_count']]


# In[118]:


d['day']=d['pickup_datetime'].dt.dayofweek


# In[119]:


d.head(5)


# In[126]:


ax=d['day'].value_counts().plot(kind='bar',cmap='magma')
for bars in ax.containers:
    ax.bar_label(bars)


# In[127]:


d.groupby('day')['passenger_count'].sum()


# Sundays are observed to be least favs for travelling

# In[ ]:




