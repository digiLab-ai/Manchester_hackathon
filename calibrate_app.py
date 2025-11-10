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
# tl.set_api_key("")

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

experiment_option = st.sidebar.selectbox(
    "Experiment:",
    ("Experiment A","Experiment B"),
)

constant = 1E9







progress = st.progress(0)
status = st.empty()
column1, column2 = st.columns(2)

if st.sidebar.button("Run Inference"):
    df_experiment = 0
    df_experiment_std = 0
    emulator_id = "Manchester_GDPS_Emulator"
    emulator = tl.Emulator(id=emulator_id)

    try:
        status.text("checking for existing emulator..." )
        df_input = emulator.view_train_data()
        scoreparams = tl.ScoreParams(metric="R2")
        R2 = np.array(emulator.score(params=scoreparams)).flatten()[0]
        emulator_exists = 1
        if R2<0.99:
            status.text("emulator exists but is not valid..." )    


        else:

            status.text(f"emulator already exists with R squared of {R2}" )
            emulator_performs = 1
    except:
        emulator_exists = 0
    if emulator_exists:

        df_obs = pd.DataFrame({'y': [0.1]})
        df_std = pd.DataFrame({'y': [0.01]})
        emulator.calibrate(df_obs, df_std)
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
