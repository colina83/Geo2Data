import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)
from functions import *
import streamlit as st

st.set_page_config(layout="wide",page_title="DS&ML SIG Houston ", page_icon="🖖")
header = st.container()

with header:
    st.title("Primary Rock Types Cluster Dashboard")
    st.subheader("Primary Rock Types for 1 well in the Williston Basin")

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

    st.subheader("Select the depth for the  Well Data")
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

        cluster_model = creating_model(df_scale, data, al, k)

    else:
        q = st.slider('Select a Quantile', min_value=0.1,
                              max_value=1.0, value=0.4, step=0.1, format="%.1f")
        st.info(f'MeanShift uses the Quantile to automatically detect the bandwith')

        cluster_model = creating_model_ms(df_scale,al,q,data)[0]


with col1:
    st.subheader(f"Plot for {al} algorithm")
    if al != "MeanShift":
        st.plotly_chart(plot_model(cluster_model,k))
    else:
        st.plotly_chart(plot_model(cluster_model,creating_model_ms(df_scale,al,q,data)[1]))
        st.info(f'The number of Clusters for the {al} algorithm is {creating_model_ms(df_scale,al,q,data)[1]} for'
                f' a {q} quantile')

with col2:
    st.subheader(f"Log plot with depths between {min} ft - {max} ft")
    st.pyplot(make_plot(cluster_model,min, max))


