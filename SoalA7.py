import streamlit as st
import pandas as pd
import numpy as np

df = pd.DataFrame(
    np.random.randn(10, 2),
    columns=['x', 'y'])
st.line_chart(df)

#bar Chart
st.area_chart(df)

#Data Chart
st.bar_chart(df)
