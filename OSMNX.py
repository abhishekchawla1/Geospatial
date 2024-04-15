#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import osmnx as ox


# In[2]:


city=ox.geocode_to_gdf('Dehradun, India')
ox.project_gdf(city).plot()


# In[3]:


box=ox.graph_from_bbox(30.3386219,30.3517506,78.0225711,78.0088431,network_type='drive')


# In[4]:


ox.plot_graph(ox.project_graph(box))


# In[5]:


box2=ox.graph_from_point((30.3517506,78.0225711),dist=750,network_type='all')
ox.plot_graph(box2)


# In[6]:


box3=ox.graph_from_address('Kaulagarh, Dehradun',network_type='drive')


# In[7]:


ox.plot_graph(box3)


# In[8]:


box4=ox.graph_from_place('Dehradun, India',network_type='drive')
ox.plot_graph(box4)


# In[9]:


box5=ox.graph_from_place('Dehradun, India',network_type='walk')


# In[10]:


stats=ox.basic_stats(box5)


# In[11]:


ox.plot_graph(box5)


# In[12]:


stats


# In[13]:


G=ox.graph_from_place('Delhi, India',network_type='drive')


# In[14]:


ox.plot_graph(G,figsize=(100,100))


# In[15]:


type(box2)


# Conversion

# In[16]:


M=ox.convert.to_undirected(G)


# In[17]:


D=ox.convert.to_digraph(G)


# In[18]:


gdf_nodes,gdf_edges=ox.graph_to_gdfs(G)


# In[19]:


ox.plot_graph(M)


# In[20]:


gdf_nodes


# In[21]:


gdf_edges


# In[22]:


gdf_nodes.plot(figsize=(50,50))


# In[23]:


gdf_edges.plot(figsize=(50,50))


# Street Stats

# In[24]:


proj=ox.project_graph(G)


# In[25]:


proj


# In[26]:


nodes_proj=ox.graph_to_gdfs(proj,edges=False)
g=nodes_proj.unary_union.convex_hull.area


# In[27]:


g


# In[28]:


ox.stats.basic_stats(G,area=None)


# ox.save_graph_geopackage(G, filepath="./data/mynetwork.gpkg")
# ox.save_graphml(G, filepath="./data/mynetwork.graphml")

# Routing

# In[31]:


G=ox.routing.add_edge_speeds(G)
G=ox.routing.add_edge_travel_times(G)


# In[32]:


origin=ox.distance.nearest_nodes(G,X=77.2333,Y=28.6665)
destination=ox.distance.nearest_nodes(G,X=77.0850,Y=28.5566)


# In[33]:


route=ox.shortest_path(G,origin,destination,weight='travel_time')


# In[35]:


fig,ax=ox.plot_graph_route(G,route,node_size=1)


# In[36]:


summary=ox.routing.route_to_gdf(G,route)


# In[37]:


summary


# In[38]:


summary['length'].sum()


# In[39]:


summary.plot()


# In[40]:


import folium


# In[41]:


m=folium.Map(location=[28.5566,77.0850],zoom_start=10)


# In[45]:


summary.iloc[0]['geometry'].__geo_interface__


# In[51]:


for index,row in summary.iterrows():
    coordinates=row['geometry'].coords
    co_list=[[c[1],c[0]] for c in coordinates]
    folium.PolyLine(co_list,popup=summary['length'].sum()/1000).add_to(m)


# In[52]:


m


# In[56]:


ox.plot_graph(ox.graph_from_place('New York, USA',network_type='walk'))


# In[58]:


summary.head(1)


# In[63]:


place='Dehradun, India'
tags={'building':True}
gdf=ox.features_from_place(place,tags)
gdf.shape


# In[65]:


bikaner=ox.graph_from_point((28.0229,73.3119),dist=1000,network_type='all')


# In[67]:


ox.plot_graph(bikaner,figsize=(100,100))


# In[69]:


gdf_nodes,gdf_edges=ox.graph_to_gdfs(bikaner)


# In[70]:


gdf_edges


# In[73]:


gdf_nodes


# In[74]:


gdf_edges.plot()


# In[75]:


G=ox.graph_from_place('Dehradun, India',network_type='drive')


# In[76]:


m1=ox.plot_graph_folium(G,popup_attribute='name',weight=2,color='#8b000')


# In[77]:


m1


# In[78]:


G.nodes()


# In[80]:


origin=list(G.nodes())[0]
destination=list(G.nodes())[-1]
route=nx.shortest_path(G,origin,destination)


# In[81]:


m2=ox.plot_route_folium(G,route,weight=10)


# In[82]:


m2


# In[83]:


m3=ox.plot_route_folium(G,route,route_map=m1,popup_attribute='length')


# In[84]:


m3


# In[85]:


tags={'leisure':'park'}
gdf=ox.features_from_place(place,tags)


# In[87]:


gdf.shape


# In[88]:


#place bounaries


# In[90]:


city=ox.geocode_to_gdf('Dehradun, India')
city_projection=ox.projection.project_gdf(city)
ax=city_projection.plot(fc='gray')


# In[93]:


gdf=ox.features_from_place('Dehradun, India',{'building':True})
ox.plot_footprints(gdf,figsize=(100,100))


# In[ ]:




