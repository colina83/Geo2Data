import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)
from functions import *
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide",page_title="DS&ML SIG Houston ", page_icon="🖖")
header = st.container()

with header:
    st.title("Primary Rock Types Cluster Dashboard")
    st.subheader("Data Analysis for 1 well in the Williston Basin")

@st.cache
def get_data():
    las = lasio.read("well_log.las")
    df = las.df()
    df_tc = df.copy()
    df_tc = df_tc[['GR','RHOZ','NPHI']]
    df_tc = df_tc[df_tc['RHOZ'].between(1.8, 3.2)]
    df_tc = df_tc[df_tc.index > 2000]
    df_tc.dropna(inplace=True)
    return df_tc

d = get_data()

#Side Bar

col1, col2 = st.columns(2)

with col1:

    st.subheader("Select the limits for your Well Data")
    basin = st.selectbox('Select a Formation', ('None - User Selection','Bakken Petroleum System'))
    if basin == 'Bakken Petroleum System':
        min = 9208
        max = 9267
        st.warning(f"You have selected a particular formation located between {min} ft.  and {max} ft.")
    else:
        min = st.number_input('Select the minimum depth',2000,10000,value=2000)
        max = st.number_input('Select the maximum depth',2000,10752,value=10752)

    data = d.copy()
    data = data.loc[min:max, :]
    df_scale = scale_data(data)

    st.subheader("Clustering")
    al = st.selectbox('Select an Unsupervised Learning Model',
                  ['K-means','Gaussian Mixture',
                   'Agglomerative Clustering','MeanShift'])

    if al == 'K-means' or al == 'Gaussian Mixture' or al == 'Agglomerative Clustering':
        k=st.slider('Select number of Clusters',2,10)

        cluster_model = creating_model(df_scale, data, al, k)[0]
        ss = creating_model(df_scale, data, al, k)[1]
        st.info(f"The silhouette score is {round(ss,2)}")
    else:

        q = st.slider('Select a Quantile', min_value=0.1,
                                  max_value=1.0, value=0.4, step=0.1, format="%.1f")
        st.info(f'MeanShift uses the Quantile to automatically detect the bandwith')

        if creating_model_ms(df_scale,al,q,data)[1] == 1 or creating_model_ms(df_scale,al,q,data)[1] > 10:
            st.error(f"The value setup in the quantile variable yields a cluster number "
                     f" of {creating_model_ms(df_scale,al,q,data)[1]}, you need a minimum of 2"
                     f" clusters and no more than 10 for this analysis - Please select another quantile value")
            st.stop()

        cluster_model = creating_model_ms(df_scale, al, q, data)[0]




with col2:
    st.subheader(f"3D Cluster Plot for {al} algorithm")
    if al != "MeanShift":
        st.plotly_chart(plot_model(cluster_model,k))
    else:
        st.plotly_chart(plot_model(cluster_model,creating_model_ms(df_scale,al,q,data)[1]))
        with col1:
            st.info(f'The number of Clusters for the {al} algorithm is {creating_model_ms(df_scale,al,q,data)[1]} for'
                f' a {q} quantile')

with col1:
    st.subheader(f"Log plot with depths between {min} ft - {max} ft")
    st.pyplot(make_plot(cluster_model,min, max))

with col2:
    st.subheader("2D Cluster Plot")

    def Plot_2D(cluster_model,k=4):
        template = {
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
        i, j = 1, k
        cluster_color_dict = {}
        for k, v in template.items():
            if int(k) >= i and int(k) <= j:
                cluster_color_dict[k] = v

        fig = px.scatter(cluster_model,
                         y='RHOZ',
                         x='NPHI',
                         color='Clusters',
                         title='Density (RHOZ) vs Neutron (NPHI)',
                         color_continuous_scale=list(cluster_color_dict.values()),width=1000, height=500)

        fig1 = px.scatter(cluster_model,
                         y='RHOZ',
                         x='GR',
                         color='Clusters',
                         title='Density (RHOZ) vs Gamma Ray (GR)',
                         color_continuous_scale=list(cluster_color_dict.values()),width=1000, height=500)

        fig2 = px.scatter(cluster_model,
                         y='NPHI',
                         x='GR',
                         color='Clusters',
                         title='Neutron (NPHI) vs Gamma Ray (GR)',
                         color_continuous_scale=list(cluster_color_dict.values()),width=1000, height=500)


        return fig,fig1,fig2

    if al != "MeanShift":
        st.plotly_chart(Plot_2D(cluster_model, k)[0])
        st.plotly_chart(Plot_2D(cluster_model, k)[1])
        st.plotly_chart(Plot_2D(cluster_model, k)[2])

    else:

        st.plotly_chart(Plot_2D(cluster_model,creating_model_ms(df_scale,al,q,data)[1])[0])
        st.plotly_chart(Plot_2D(cluster_model, creating_model_ms(df_scale,al,q,data)[1])[1])
        st.plotly_chart(Plot_2D(cluster_model, creating_model_ms(df_scale,al,q,data)[1])[2])


