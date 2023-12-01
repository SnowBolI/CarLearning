import streamlit as st

st.checkbox('yes')
st.button('Click')

st.radio('Pick your gender', ['Male', 'Female'])

st.selectbox('Pick your Gender', ['Male', 'Female'])

st.multiselect('Choose a planet', ['Mars','Venus','Earth'])

st.select_slider('Pick a mark', options=['Bad','Good', 'Excellent'])

st.slider('Pick a number ', min_value=0, max_value=50)
