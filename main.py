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
#from matplotlib import pyplot
#from PyCrystallography import unit_cell
#from PyCrystallography import lattice
import os

import streamlit as st

st.set_page_config(
    page_title="Multipage App",
    #page_icon="",
)

image = Image.open('./images/large.PNG')
new_img = image.resize((180, 100))
#st.image(new_img)
left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image(new_img)


st.title("Main Page")
st.sidebar.success("Select a page above.")

st.markdown('###### Upload two .txt patterns separately and let them be calculated.')

# Allow only .csv and .xlsx files to be uploaded
uploaded_file = st.file_uploader("Upload Main XRD Pattern", type=["txt"])

name = uploaded_file
if not name:
  st.warning('Please input a .txt file.')
  st.stop()
st.success('Done.')



uploaded_file2 = st.file_uploader("Upload Compairing XRD Pattern", type=["txt"])

name2 = uploaded_file2
if not name2:
  st.warning('Please input a .txt file.')

  st.stop()
st.success('Done.')

cif_file = st.file_uploader("Upload CIF file", type=["cif","CIF"],
                             help="Any standard CIF including ICSD, COD, CCDC exports")

name3 = cif_file
if not name3:
  st.warning('Please input a .cif file.')

  st.stop()
st.success('Done.')


df = pd.read_fwf(name)
df.to_csv('First.csv', index=False)

np.savetxt('First.xy', df, fmt='%f', delimiter='\t')
np.savetxt('First.csv', df, fmt='%f', delimiter=',')

df = pd.read_fwf(name2)
df.to_csv('Second.csv', index=False)
np.savetxt('Second.csv', df, fmt='%f', delimiter=',')




#os.system("WH.py")


#global weather1
#global weather2
#global weather3

weather1 = pd.read_csv('First.csv', names=['2Theta','Int'])
weather2 = pd.read_csv('Second.csv', names=['2Theta','Int2'])

weather3 = weather1-weather2
np.savetxt('testTheta2.txt', weather3, fmt='%f')

#weather3 = pd.read_csv('testTheta2.txt', names=['2Theta','Diff'])
#weather5 = pd.read_csv('WH-realx.txt', names=['sinTheta'])
#weather6 = pd.read_csv('WH-realy.txt', names=['BetaCosTheta'])

#zett = np.polyfit(weather5['sinTheta'], weather6['BetaCosTheta'], 1)
#pee = np.poly1d(zett)

#weather_WH = weather5.join(weather6)
#print(weather_WH)
#weather_avg = sum(weather_WH['BetaCosTheta']) / len(weather_WH['BetaCosTheta'])
#print(weather_avg)


#integral = np.array([])
#for col in weather1.columns[1:]:
#    temp = weather1.iloc[:, 1:].apply(lambda x: integrate.trapz(x,weather1['Int']))
#    integral = np.append(integral,temp)


#integral2 = np.array([])
#for col in weather2.columns[1:]:
#    temp = weather2.iloc[:, 1:].apply(lambda x: integrate.trapz(x,weather2['Int2']))
#    integral2 = np.append(integral2,temp)


#df_app = 'crash1', 'crash2'

#df_merged = [integral, integral2]

#np.savetxt('area.txt', df_merged, fmt='%f', delimiter=',')
#area = pd.read_csv('area.txt', names=['Area','Sample'])

#np.savetxt('area2.txt', df_merged, fmt='%f', delimiter=',')
#area = pd.read_csv('area2.txt', names=['Area2','Sample'])


#area['Sample'] = df_app

#columns_titles = ["Sample","Area"]
#area=area.reindex(columns=columns_titles)

#area.to_csv("Area.csv", index=False)

#stocks = pd.read_csv('Area.csv')
#stocks.to_csv("Area.csv", index=False)




#dfheat = pd.read_csv('crash1.csv', names=['2Theta','Int'])
#df2heat = pd.read_csv('crash2.csv', names=['2Theta','Int2'])
#dfheat['Int2'] = df2heat['Int2']

#df = pd.read_fwf('FWHMFirst.txt', header=None)

#np.savetxt('FHWMFirstSecond.csv', df, fmt='%s', delimiter=' ')

#data = pd.read_csv('FHWMFirstSecond.csv', sep=" ", names=['Int','Scherrer'])