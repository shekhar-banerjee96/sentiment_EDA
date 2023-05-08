import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

df = sns.load_dataset("penguins")
fig,_ = sns.barplot(data=df, x="island", y="body_mass_g")

st.pyplot(fig)