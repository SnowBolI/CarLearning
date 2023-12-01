import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

st.title('Prediksi Harga Mobil')
st.image('./herta.gif')
st.header("Dataset")
#open file csv
df1 = pd.read_csv('CarPrice.csv')

brands = st.sidebar.multiselect('Show cars for brands', df1['CarName'].unique())
df2 = df1[df1['CarName'].isin(brands)]
st.dataframe(df2)

st.write("Grafik Highway-mpg")
chart_highwaympg = pd.DataFrame(df1, columns=["highwaympg"])
st.line_chart(chart_highwaympg)

st.write("Grafik curbweight")
chart_curbweight = pd.DataFrame(df1, columns=["curbweight"])
st.line_chart(chart_curbweight)

st.write("Grafik horsepower")
chart_horsepower = pd.DataFrame(df1, columns=["horsepower"])
st.line_chart(chart_horsepower)

#input nilai dari variable independent
highwaympg = st.sidebar.number_input("Highway ", 0,10000000)
curbweight = st.sidebar.number_input("Curbwight ", 0,10000000)
horsepower = st.sidebar.number_input("horsepower  ", 0,10000000)

if st.button('Prediksi'):
    try:
        #prediksi variable yang telah diinputkan 
        car_prediction = model.predict([[highwaympg, curbweight, horsepower]])

        # convert ke string
        harga_mobil_str = np.array(car_prediction)
        harga_mobil_float = float(harga_mobil_str[0])

        #tampilkan hasil prediksi
        harga_mobil_formatted = "{:,.2f}".format(harga_mobil_float)
        st.markdown(harga_mobil_formatted)
    except Exception as e:
        st.write("An error occurred: ", str(e))
