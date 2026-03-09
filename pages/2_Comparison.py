import streamlit as st
import pandas as pd
import plost
import base64
import numpy as np
#import Main
#from numpy import trapz
from scipy.integrate import simpson
from scipy.integrate import trapezoid
from scipy import*

# Contents of ~/my_app/main_page.py


from PIL import Image



im = 'images/favicon.png'
st.set_page_config(
    page_title="FellX v0.8",
    page_icon=im,
    layout="wide",
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("## ")
st.sidebar.markdown("# ")

#st.sidebar.subheader('Heat map parameter')
#time_hist_color = st.sidebar.selectbox('Color by', '')
#st.sidebar.subheader('Donut chart parameter')
#donut_theta = st.sidebar.selectbox('Select data', ('Theta', 'Area2'))

weather1 = pd.read_csv('First.csv', names=['2Theta','Int'])
weather2 = pd.read_csv('Second.csv', names=['2Theta','Int2'])


integral = np.array([])
for col in weather1.columns[1:]:
    temp = weather1.iloc[:, 1:].apply(lambda x: integrate.trapezoid(x,weather1['Int']))
    integral = np.append(integral,temp)
#print(integral)
#np.savetxt('Integration1.txt', integral, fmt='%f')

integral2 = np.array([])
for col in weather2.columns[1:]:
    temp = weather2.iloc[:, 1:].apply(lambda x: integrate.trapezoid(x,weather2['Int2']))
    integral2 = np.append(integral2,temp)


df_app = 'First', 'Second'
df_merged = [integral, integral2]
np.savetxt('area.txt', df_merged, fmt='%f', delimiter=',')

area = pd.DataFrame()
area['Sample'] = df_app

columns_titles = ["Sample","Area"]
area=area.reindex(columns=columns_titles)
#print(area)
area.to_csv("Area.csv", index=False)

stocks = pd.read_csv('Area.csv')
stocks.to_csv("Area.csv", index=False)




dfheat = pd.read_csv('First.csv', names=['\u00b0 2Theta','Int'])
stocks = pd.read_csv('Area.csv')

c1, c2 = st.columns((3,1))
with c1:
    #st.markdown('#### Main XRD pattern')
    plost.scatter_hist(
        data=dfheat,
        x='\u00b0 2Theta',
        y='Int',
        size='Int',
        color='Int',
        opacity=0.5,
        aggregate='count',
        width=200,
        height=200,
        legend='bottom',
        use_container_width=True
    )
    
    st.markdown('#### Data Main XRD')
    st.markdown('###### Check if all fields match')
    plost.xy_hist(
        data=dfheat,
        x='\u00b0 2Theta',
        y='Int',
        #x_bin=100,
        #y_bin='Int2',
        use_container_width=True,
    )
    

with c2:
    st.markdown('### XRD patterns vs. in %')
    plost.donut_chart(
        data=stocks,
        #theta=donut_theta,
        theta='Area',
        color='Sample',
        
        legend='bottom', 
        use_container_width=True)
