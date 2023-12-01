import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.sidebar.selectbox('Select page', ['Home', 'Landing', 'Dashboard'])
# Import Image
st.image('./herta.gif')

# Import dataset
df = pd.DataFrame({
    'first column' : [1,2,3,4],
    'second column' : [10,20,30,40] 
    })
st.write('Dataset :', df)

data = pd.read_csv("kry.csv")
ds = pd.DataFrame(data)
st.bar_chart(ds)
