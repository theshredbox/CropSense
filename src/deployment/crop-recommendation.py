import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from PIL import Image
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
import joblib


page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-color:black;

}}
[data-testid="stSidebar"] {{
background-color:black;

}}
[data-testid="stHeader"] {{
background-color:blue;
}}
[data-testid="stToolbar"] {{
background-color:black;

}}
</style>
"""




logo_image = Image.open('loo.png')

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    st.image(logo_image, width=350, use_column_width=False, clamp=True)


st.markdown(
        """
        <style>
                .appview-container .main .block-container {{
                background:url("https://images.unsplash.com/photo-1493673272479-a20888bcee10?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80.jpg") no-repeat;
                background-size:cover;
                width:100vw !important;
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=1, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )


def load_bootstrap():
        return st.markdown("""<link rel="stylesheet" 
        href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
        integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
        crossorigin="anonymous">""", unsafe_allow_html=True)

with st.sidebar:
    
    load_bootstrap()
    st.markdown("""<h4 style='text-align: center; background-color:: black;'>
    OTHER SERVICES </h4>""",unsafe_allow_html=True)
    st.markdown(f"""<h4 style='text-align: center;margin-top:20px; color: white;>'</h4>""",
    unsafe_allow_html=True)
    st.markdown(f"""<h4 style='text-align: center; color: black;'>
     <<a style='text-align: center; color: black;' 
     type="button" class="btn btn-warning btn-lg"
     href = "http://10.244.20.156:8501/">Crop disease detection</a></h4>""", unsafe_allow_html=True)
    st.markdown("""<h5 style='text-align: center;margin-top:80px; background-color:: green;'>
    ABOUT US </h5>""",unsafe_allow_html=True)



colx, coly, colz = st.columns([1,4,1], gap = 'medium')
with coly:
    st.markdown("""
  
    
      <h4 style='text-align: center;font-family:arial;color:white;margin-top:10px'>
        Agriculture is a vital occupation practiced worldwide.
        It plays a crucial role in a country's development,
        with a significant portion of land dedicated to it.
        Adopting new agricultural technologies is essential.
        By using CropSense, farmers can generate crop recommendations based
        on factors like soil conditions (i.e) the N,P,K value and other
        conditions such as rainfall, temperature, humidity and Ph value.
        These recommendations enable informed decision-making and
        improve agricultural practices.
      </h4>
  
        """, unsafe_allow_html=True)

df = pd.read_csv('Crop_recommendation.csv')

rdf_clf = joblib.load('whybro.pkl')

X = df.drop('label', axis = 1)
y = df['label']

df_desc = pd.read_csv('Crop_Desc.csv', sep = ';', encoding = 'utf-8', encoding_errors = 'ignore')

st.markdown("<h4 style='text-align: center;color:white;'>Importance of each Feature in the Model:</h4>", unsafe_allow_html=True)


importance = pd.DataFrame({'Feature': list(X.columns),
                   'Importance(%)': rdf_clf.feature_importances_}).\
                    sort_values('Importance(%)', ascending = True)
importance['Importance(%)'] = importance['Importance(%)'] * 100

colx, coly, colz = st.columns([1,4,1], gap = 'medium')
with coly:
    color_discrete_sequence='#32612d'
    fig = px.pie(importance, values='Importance(%)', names='Feature', 
    color_discrete_sequence=[color_discrete_sequence])

    fig.update_traces(marker=dict(line=dict(color='#ffffff', width=2)))

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    st.plotly_chart(fig, use_container_width=True)   
st.markdown("<h5 style='text-align: center;color:white;'>Here you can insert the values! Our system will predict the best crop to plant!</h5>", unsafe_allow_html=True)

col1, col2 ,col3 ,col4 ,col5 ,col6 ,col7 = st.columns([1,1,4,1,4,1,1], gap = 'medium')
    
with col3:
    st.markdown('<style>.col3-input input { background-color: lightgray; }</style>', unsafe_allow_html=True)
    st.markdown('<h5 style="color: white;">Insert N (kg/ha) value:</h5>', unsafe_allow_html=True)
    n_input = st.number_input("", min_value=0, max_value=140, help='Insert here the Nitrogen density (kg/ha) from 0 to 140.', key='col3-n_input')
    st.markdown('<h5 style="color: white;">Insert P (kg/ha) value:</h5>', unsafe_allow_html=True)
    p_input = st.number_input("", min_value=5, max_value=145, help='Insert here the Phosphorus density (kg/ha) from 5 to 145.', key='col3-p_input')
    st.markdown('<h5 style="color: white;">Insert K (kg/ha) value:</h5>', unsafe_allow_html=True)
    k_input = st.number_input("", min_value=5, max_value=205, help='Insert here the Potassium density (kg/ha) from 5 to 205.', key='col3-k_input')
    st.markdown('<h5 style="color: white;">Insert Avg Temperature (ºC) value:</h5>', unsafe_allow_html=True)
    temp_input = st.number_input("", min_value=9., max_value=43., step=1., format="%.2f", help='Insert here the Avg Temperature (ºC) from 9 to 43.', key='col3-temp_input')
    st.markdown('<style>.stNumberInput input { background-color: #32612d; }</style>', unsafe_allow_html=True)
 

with col5:
    st.markdown('<style>.col5-input input { background-color: lightgray; }</style>', unsafe_allow_html=True)
    st.markdown('<h5 style="color: white;">Insert Avg Humidity (%) value:</h5>', unsafe_allow_html=True)
    hum_input = st.number_input(" ", min_value=15., max_value=99., step=1., format="%.2f", help='Insert here the Avg Humidity (%) from 15 to 99.', key='col5-hum_input')
    st.markdown('<h5 style="color: white;">Insert pH value:</h5>', unsafe_allow_html=True)
    ph_input = st.number_input(" ", min_value=3.6, max_value=9.9, step=0.1, format="%.2f", help='Insert here the pH from 3.6 to 9.9', key='col5-ph_input')
    st.markdown('<h5 style="color: white;">Insert Avg Rainfall (mm) value:</h5>', unsafe_allow_html=True)
    rain_input = st.number_input(" ", min_value=21.0, max_value=298.0, step=0.1, format="%.2f", help='Insert here the Avg Rainfall (mm) from 21 to 298', key='col5-rain_input')
   
    st.markdown('<style>.stNumberInput input { color: white; }</style>', unsafe_allow_html=True)

   



predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input]]

col5 = st.columns([3, 1, 3], gap='medium')


with col5[1]:
    st.markdown("""
        <style>
        .col5-input button {
            background-color: blue;
            color: green;
            display: block;
            margin: 0 auto;
            font-size:150px;
        }
        </style>
    """, unsafe_allow_html=True)
    predict_btn = st.button('Get Your Recommendation!', key='col5-predict_btn')


cola,colb,colc = st.columns([2,10,2])
if predict_btn:
    color_discrete_sequence = '#32612d'
    rdf_predicted_value = rdf_clf.predict(predict_inputs)
    #st.text('Crop suggestion: {}'.format(rdf_predicted_value[0]))
    with colb:
        st.markdown(f"<h3 style='text-align: center;color:white'>Best Crop to Plant: {rdf_predicted_value[0]}.</h3>", 
        unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3,1,3])
    with col2:
        df_desc = df_desc.astype({'label':str,'image':str})
        df_desc['label'] = df_desc['label'].str.strip()
        df_desc['image'] = df_desc['image'].str.strip()
        

        df_pred_image = df_desc[df_desc['label'].isin(rdf_predicted_value)]
        df_image = df_pred_image['image'].item()
        
        st.markdown(f"""<h5 style = 'text-align: center; height: 300px; object-fit: contain;'> {df_image} </h5>""", unsafe_allow_html=True)
        

    
    st.markdown(f"""<h5 style='text-align: center;font-family:verdana;'>Overall Summary about {rdf_predicted_value[0]} 
            NPK and Weather Conditions values are listed below.</h5>""", unsafe_allow_html=True)
    df_pred = df[df['label'] == rdf_predicted_value[0]]
    st.dataframe(df_pred.describe(), use_container_width = True)        
    

    
    

    

    
    

    