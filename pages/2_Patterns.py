import streamlit as st
import pandas as pd
#import plost
import base64
import numpy as np
#from scipy.integrate import simpson
#from numpy import trapz
from PIL import Image
#import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot
#from PyCrystallography import unit_cell
#from PyCrystallography import lattice
import os
import altair as alt
import streamlit as st
from main import cif_file2

im = 'images/favicon.png'
st.set_page_config(
    page_title="FellX v0.8",
    page_icon=im,
    layout="wide",
)




with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('')

plot_height = st.sidebar.slider('Specify plot height', 200, 1000, 250)
#st.button("Next Page")
st.sidebar.markdown('''
---

''')
#st.markdown('### XRD Charts')


user_input = st.number_input("Please enter a starting number for  \u00b0 2Theta. Minimum should be at least the first higher number \u00b0 2Theta data point of the two diffraction patterns, so they can adjust.")


df1 = pd.read_csv('First.csv', names=['Theta','Int'])#, skiprows = 80
df2 = pd.read_csv('Second.csv', names=['Theta2','Int2'])


#df1zone = df1['Int']
#df2zone = df2['Int2']
#df1Theta = df1['Theta']
global failure_count
global failure_count2

StartingValue = user_input 

values = df1['Theta']  # list of values

failure_count = ["fail" for x in values if x < StartingValue].count("fail")  # list comprehension

print(failure_count)

values2= df2['Theta2']  # list of values

failure_count2 = ["fail" for x in values2 if x < StartingValue].count("fail")  # list comprehension

print(failure_count2)


df1 = pd.read_csv('First.csv', names=['Theta','Int'], skiprows = failure_count)

global dfSize
#df1 = dfSize

df2 = pd.read_csv('Second.csv', names=['Theta2','Int2'], skiprows = failure_count2)
#print(df1)

weatherTheta2 = df2['Theta2']

df1zone = df1['Int']
df2zone = df2['Int2']
np.savetxt('test1zone.txt', df1zone, fmt='%f', delimiter=',')
np.savetxt('test2zone.txt', df2zone, fmt='%f', delimiter=',')


df1Theta = df1['Theta']
np.savetxt('testTheta.txt', df1Theta, fmt='%f', delimiter=',')
#np.savetxt('testInt.txt', df1zone, fmt='%f', delimiter=',')
#np.savetxt('testInt2.txt', df2zone, fmt='%f', delimiter=',')
df1 = pd.read_csv('test1zone.txt', names=['Int'])
df2 = pd.read_csv('test2zone.txt', names=['Int'])

dfmess = df1 - df2
dfmess.dropna(how='any', inplace=True)
np.savetxt('testInt10.txt', dfmess, fmt='%f', delimiter=',')

#dfTheta = pd.read_csv('testTheta.txt', header=None)
df1Theta.columns = ['Theta']
#df = pd.read_csv('testInt10.txt', header=None)
dfmess.columns = ['Int']

dfTheta = pd.read_csv('testTheta.txt', names=['2Theta'])

dfmerge = dfTheta.join(dfmess)
#print(df1Theta)
#df_merged = pd.concat([df, dfmess], ignore_index=True) #WORKK
np.savetxt('DEF.txt', dfmerge, fmt='%f', delimiter=',')








Chart1 = pd.read_csv('First.csv', names=['\u00b0 2Theta','Int'], skiprows=failure_count)

st.markdown('##### Main')
st.line_chart(Chart1, x = '\u00b0 2Theta', y = 'Int', height = plot_height)


st.markdown('##### Compairing')
Chart2 = pd.read_csv('Second.csv', names=['\u00b0 2Theta','Int'], skiprows=failure_count)
st.line_chart(Chart2, x = '\u00b0 2Theta', y = 'Int', height = plot_height)

st.markdown('##### Main - Compairing')
Chart3 = pd.read_csv('DEF.txt', names=['\u00b0 2Theta','Int'], skiprows=failure_count)
st.line_chart(Chart3, x = '\u00b0 2Theta', y = 'Int', height = plot_height)


Theta = dfTheta['2Theta']
Chart4 = Chart1['Int']
Chart5 = Chart2['Int']

Chartlog1 = np.log(Chart4)
Chartlog2 = np.log(Chart5)

np.savetxt('testLog.txt', Chartlog1, fmt='%f', delimiter=',')
np.savetxt('testLogComp.txt', Chartlog2, fmt='%f', delimiter=',')
#np.savetxt('testTheta.txt', weatherTheta, fmt='%f', delimiter=',')
#np.savetxt('testTheta.txt', weatherTheta2, fmt='%f', delimiter=',')

weatherLogXX = pd.read_csv('testLog.txt', names=['Int'])
weatherLogYY = pd.read_csv('testLogComp.txt', names=['Int'])
weatherThetaXX = pd.read_csv('testTheta.txt', names=['\u00b0 2Theta'])
weatherThetaYY = pd.read_csv('testTheta.txt', names=['\u00b0 2Theta'])

weatherMerge = weatherThetaXX.join(weatherLogXX)
weatherMerge2 = weatherThetaYY.join(weatherLogYY)

df1 = pd.DataFrame(weatherLogXX)
df2 = pd.DataFrame(weatherLogYY)


weatherLogDiff10 = weatherLogXX['Int'] - weatherLogYY['Int']

weatherLogDiff10.columns = ['Int']

weatherLogDiff12 = pd.DataFrame(weatherLogDiff10)
weatherLogDiff12.columns = ['Int']
weatherThetaXX['Int'] = weatherLogDiff12



st.markdown('##### Main - Log Scale')
st.line_chart(weatherMerge, x = '\u00b0 2Theta', y = 'Int', height = plot_height)
st.markdown('##### Comp - Log Scale')
st.line_chart(weatherMerge2, x = '\u00b0 2Theta', y = 'Int', height = plot_height)
st.markdown('##### Main - Comp - Log Scale')
st.line_chart(weatherThetaXX, x = '\u00b0 2Theta', y = 'Int', height = plot_height)

chart_data = pd.DataFrame(
    weatherThetaXX,
    columns=['a', 'b'])

c = alt.Chart(chart_data).mark_circle().encode(
    x='a', y='b', size='b', color='b', tooltip=['a', 'b'])

st.altair_chart(c, use_container_width=True)