import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import os

import twinlab as tl
tl.set_api_key("")

import plotly.express as px
colors = ["#16425B","#16D5C2","#EBF38B"]
# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Inference Dashboard", layout="wide")

st.title("Inference Dashboard")

# -------------------------------
# Sidebar - Simulation Settings
# -------------------------------
st.sidebar.image("digiLab_logo.png",  use_container_width=True)
st.sidebar.header("Calibration Settings")

X1_name = "thickness"
X1_units = "mm"
X2_name = "solubility"
X2_units = "au"
X3_name = "diffusivity"
X3_units = "1E19"
# Data generation parameters


option = st.sidebar.selectbox(
    "Inference Quantity",
    (X1_name,X2_name,X3_name),
)


constant = 1E9
# replace with galaxy TMAP workflow
def run_simulator(i):
    df = pd.read_csv("Data//tmap-active-loop_training_data.csv")
    df_input = pd.DataFrame({
    "solubility": df["solubility"][:batch_size*(i+1)],
    "diffusivity":constant*(df["diffusivity"][:batch_size*(i+1)]),
    "thickness": df["thickness"][:batch_size*(i+1)],
    "output_flow":df["output_flow"][:batch_size*(i+1)]
    })
    return df_input






progress = st.progress(0)
status = st.empty()
column1, column2 = st.columns(2)

if st.sidebar.button("Run Inference"):
    column1.image("calibration.gif",  use_container_width=True)
    # st.session_state["data"] = generate_data(n_samples, x_min, x_max, noise)

# Display data
# if "data" in st.session_state:
#     st.subheader("📊 Generated Data")
#     df = st.session_state["data"]
#     st.write(df.head())
#     fig, ax = plt.subplots()
#     ax.scatter(df["X"], df["y"], alpha=0.6)
#     ax.set_xlabel("X")
#     ax.set_ylabel("y")
#     st.pyplot(fig)
# else:
#     st.info("Generate data to begin.")

# -------------------------------
# Sidebar - Model Settings
# -------------------------------
# 

# -------------------------------
# Training Section
# -------------------------------
