import streamlit as st

st.header('This is the app title')

if st.button('Say Hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')