import streamlit as st
import datetime

st.number_input("Enter a number", value=1)

st.text_input('Email Address')

st.date_input('Traveling Date')

st.time_input('School time', datetime.time(8, 00))

st.text_area('description')

st.file_uploader("Upload a Photo")

st.color_picker('Choose your favourite colour', '#f200ff')
