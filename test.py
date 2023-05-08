import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig , ax = plt.subplots()
df = sns.load_dataset("penguins")
sns.countplot(data = df,x = 'island')
st.pyplot(fig)