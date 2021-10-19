import pandas as pd
from sklearn.cluster import KMeans,MeanShift, estimate_bandwidth,AgglomerativeClustering
pd.set_option('display.float_format', '{:.2f}'.format)
import numpy as np
import plotly.express as px
import lasio
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler




## Create a model
def creating_model(df_scale, data, al, k=1):
    if al == 'K-means':
        cluster_model = KMeans(n_clusters=k, random_state=42)
        cluster_model.fit(df_scale)
        df_tc_cluster = data.copy()
        df_tc_nrm_cluster = df_scale.copy()
        df_tc_cluster['Clusters'] = cluster_model.labels_ + 1
        df_tc_nrm_cluster['Clusters'] = cluster_model.labels_ + 1
        return df_tc_cluster

    elif al == 'Gaussian Mixture':
        EM = GaussianMixture(n_components=k)
        EM.fit(df_scale)
        cluster_model = EM.predict(df_scale)
        df_tc_cluster = data.copy()
        df_tc_cluster['Clusters'] = cluster_model + 1
        df_scale['Clusters'] = cluster_model
        return df_tc_cluster

    elif al == 'Agglomerative Clustering':
        hac = AgglomerativeClustering(n_clusters=k, affinity="euclidean",
                                      linkage='ward')
        hac.fit(df_scale)
        df_tc_cluster = data.copy()
        df_tc_cluster['Clusters'] = hac.labels_ + 1

        return df_tc_cluster

def creating_model_ms(df_scale, al, q, data):
    if al == "MeanShift":
        bw = estimate_bandwidth(df_scale, quantile=q, n_samples=1000)
        ms = MeanShift(bandwidth=bw, bin_seeding=True)
        model = ms.fit(df_scale)
        labels = model.labels_ + 1
        cluster_centers = model.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        df_tc_cluster = data.copy()
        df_tc_cluster["Clusters"] = labels

        return df_tc_cluster,n_clusters_


### CLuster Plot
def plot_model(cluster_model,k):
    template = {
        1:'yellow',
        2:'CornflowerBlue',
        3:'orange',
        4:'green',
        5:'black',
        6:'fuchsia',
        7:'red',
        8:'aqua',
        9:'deepskyblue',
        10:'greenyellow'
    }
    i,j = 1,k
    cluster_color_dict = {}
    for k,v in template.items():
        if int(k) >= i and int(k) <= j:
            cluster_color_dict[k] = v


    fig = px.scatter_3d(cluster_model,
                    x='NPHI',
                    y='RHOZ',
                    z='GR',
                    color='Clusters',
                    color_continuous_scale=list(cluster_color_dict.values()))

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_traces(marker_size=3)
    return fig



def make_plot(well_df, top_depth, bottom_depth):
    track_fig, ax = plt.subplots(figsize=(12, 9))

    #Set up the plot axes
    ax1 = plt.subplot2grid((1, 5), (0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((1, 5), (0, 1), rowspan=1, colspan=1, sharey=ax1)
    ax3 = ax2.twiny() #Twins the y-axis for the density track with the neutron track

    # As our curve scales will be detached from the top of the track,
    # this code adds the top border back in without dealing with splines
    ax10 = ax1.twiny()
    ax10.xaxis.set_visible(False)
    ax11 = ax2.twiny()
    ax11.xaxis.set_visible(False)

    # Gamma Ray track
    ax1.plot(well_df["GR"], well_df.index, color="black", linewidth=0.5)
    ax1.set_xlabel("Gamma")
    ax1.xaxis.label.set_color("black")
    ax1.set_xlim(0, 200)
    ax1.set_ylabel("Depth (m)")
    ax1.tick_params(axis='x', colors="black")
    ax1.spines["top"].set_edgecolor("black")
    ax1.title.set_color('black')
    ax1.set_xticks([0, 50, 100, 150, 200])

    left_col_value = 0
    right_col_value = 200
    span = abs(left_col_value - right_col_value)

    cmap = plt.get_cmap('inferno_r')
    color_index = np.arange(left_col_value, right_col_value, span / 5)

    # Loop through each value in the color_index
    for index in sorted(color_index):
        index_value = (index - left_col_value) / span
        color = cmap(index_value) #obtain colour for color index value
        ax1.fill_betweenx(well_df.index, well_df["GR"] , 200, where=well_df["GR"] >= index,  color=color)

    # Density track
    ax2.plot(well_df["RHOZ"], well_df.index, color="green", linewidth=0.5)
    ax2.set_xlabel("Density")
    ax2.set_xlim(1.95, 2.95)
    ax2.xaxis.label.set_color("green")
    ax2.tick_params(axis='x', colors="green")
    ax2.spines["top"].set_edgecolor("green")
    ax2.set_xticks([1.95, 2.45, 2.95])

    # Neutron track placed on top of density track
    ax3.plot(well_df["NPHI"], well_df.index, color="blue", linewidth=0.5)
    ax3.set_xlabel('Neutron')
    ax3.xaxis.label.set_color("blue")
    ax3.set_xlim(0.4, -0.1)
    ax3.tick_params(axis='x', colors="blue")
    ax3.spines["top"].set_position(("axes", 1.08))
    ax3.spines["top"].set_visible(True)
    ax3.spines["top"].set_edgecolor("blue")
    ax3.set_xticks([0.4, 0.1, -0.1])

    x1 = well_df['RHOZ']
    x2 = well_df['NPHI']

    x = np.array(ax2.get_xlim())
    z = np.array(ax3.get_xlim())

    nz = ((x2 - z.max()) / (z.min() - z.max())) * (x.max() - x.min()) + x.min()

    ax2.fill_betweenx(well_df.index, x1, nz, where=x1 >= nz, interpolate=True, color='lightgrey')
    ax2.fill_betweenx(well_df.index, x1, nz, where=x1 <= nz, interpolate=True, color='yellow')

    # Cluster Track
    if any("Clusters" in s for s in well_df.columns.tolist()):
        log = [l for l in well_df.columns.tolist() if "Clusters" in l][0]
        str(log)
        ax4 = plt.subplot2grid((1,5), (0,2), rowspan=1, colspan=1, sharey=ax1)
        ax15 = ax4.twiny()
        ax15.xaxis.set_visible(False)

        ax4.plot(well_df.index, well_df[log], color="black", linewidth=0.5)
        ax4.set_xlabel("Lithology")
        ax4.xaxis.label.set_color("black")
        ax4.set_xlim(0, 1)
        ax4.tick_params(axis='x', colors="black")
        ax4.spines["top"].set_edgecolor("black")
        ax4.title.set_color('black')

        cluster_color_dict = {
                        1: 'yellow',
                        2: 'CornflowerBlue',
                        3: 'orange',
                        4: 'green',
                        5: 'black',
                        6: 'fuchsia',
                        7: 'red',
                        8: 'aqua',
                        9: 'deepskyblue',
                        10: 'greenyellow'
                            }


        # Loop through each value in the color_index
        for k, v in cluster_color_dict.items():
            color = v #obtain colour for color index value
            ax4.fill_betweenx(well_df.index, 10, where=well_df[log] >= k,  color=color)

        ax4.set_xticks([0, 1])
        ax4.set_ylim(bottom_depth, top_depth)
        ax4.grid(which='major', color='lightgrey', linestyle='-')
        ax4.xaxis.set_ticks_position("top")
        ax4.xaxis.set_label_position("top")
        ax4.spines["top"].set_position(("axes", 1.02))

        plt.setp(ax4.get_yticklabels(), visible=False)

    # Misc Formatting
    for ax in [ax1, ax2]:
        ax.set_ylim(bottom_depth, top_depth)
        ax.grid(which='major', color='lightgrey', linestyle='-')
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.spines["top"].set_position(("axes", 1.02))

    for ax in [ax2]:
        plt.setp(ax.get_yticklabels(), visible=False)

    plt.tight_layout()
    track_fig.subplots_adjust(wspace=0.25)


    return track_fig

def get_data():
    las = lasio.read("well_log.las")
    df = las.df()
    df_tc = df.copy()
    df_tc = df_tc[['GR','RHOZ','NPHI']]
    df_tc = df_tc[df_tc['RHOZ'].between(1.8, 3.2)]
    df_tc = df_tc[df_tc.index > 2000]
    df_tc.dropna(inplace=True)
    return df_tc

def scale_data(df):
    scaler = StandardScaler()
    df_tc_nrm = scaler.fit_transform(df)
    df_tc_nrm = pd.DataFrame(df_tc_nrm, columns=df.columns, index=df.index)
    return df_tc_nrm



df_scale = scale_data(get_data())

model = KMeans()
visualizer = KElbowVisualizer(model,k=(2,30), timings = True)
visualizer.fit(df_scale)
visualizer.show()



'''
###### Preparing the data
scaler = StandardScaler()
df_tc_nrm = scaler.fit_transform(df_tc)

# Create DataFrame
df_tc_nrm = pd.DataFrame(df_tc_nrm, columns=df_tc.columns, index=df_tc.index)

# Inspect
print(df_tc_nrm.shape)
df_tc_nrm.describe().T

def create_model

'''
















