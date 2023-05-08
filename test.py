import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig , ax = plt.subplots()
df = sns.load_dataset("penguins")
sns.barplot(data=df, x="island", y="body_mass_g",ax=ax)

st.pyplot(fig)