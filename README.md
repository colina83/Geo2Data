# Geo2Data
Geo2Data Workshop - Visualizing Data in Python – Oil and Gas Applications

This dashboard uses log well data from 1 well in the Williston Basin, the idea is to use the data for the following physical properties 'GR','RHOZ','NPHI' in order to generate cluster of data 
with similar characteristics, then by applying different unsupervised clustering algorithms we can establish relationships such that each cluster represents a rock type.

Although the dataset contains 55 logs, we will concentrate in the following measurements:
1. Gamma Ray (GR): Tipically relates to Lithology

2. Density (RHOZ): Related to bulk density (RHOB)

3. Neutron (NPHI): Related to porosity


In the context of this project, we will evaluate if different rock types are discernable with an unsupervised learning algorithm, now in practice, this is far more 
complex but the idea is that this approach will provide some guidance regarding the relationship between the clustered logs and the rock types

The main goal of this dashboard is to try to answer the question, which is the best clustering algorithm to use in the Williston Basin?, now the solution depends on the characteristics of the data,
and as such, the first parameter to select in the dashboard is the depth, this will select the data to a particular location. I have encoded the location of 
the Bakken Petroleum System to assist with the exploratory data analysis.

The dashboard allows you to choose the number of estimators, meaning the number of clusters that the algorithm will segment the data, and each cluster will be assing a distinct color. The user can select one of the following algorithms:

- K-Means
- Gaussian Mixture
- Agglomerative Clustering
- MeanShift

Lastly, the dashboard provides 2 visualizations; a 3D model with the 3 variables (Gamma Ray (GR),Density (RHOZ) &
Neutron (NPHI)), since there can be variations between each of the variables, you can explore a 2D plot to locate differences between the clustering algorithms

App hosted by Streamlit:

https://share.streamlit.io/colina83/geo2data/main.py




