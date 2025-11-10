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
import plotly.graph_objects as go
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


def update_experiment_plot():

    # Parameters
    r = 1.0
    h = 0.2
    color = "lightblue"
    n = 60

    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    x_top, y_top, z_top = x, y, np.full_like(x, h/2)
    x_bottom, y_bottom, z_bottom = x, y, np.full_like(x, -h/2)

    X = np.concatenate([x_top, x_bottom])
    Y = np.concatenate([y_top, y_bottom])
    Z = np.concatenate([z_top, z_bottom])

    i, j, k = [], [], []

    # --- Top face ---
    for t in range(1, n - 1):
        i.append(0)
        j.append(t)
        k.append(t + 1)

    # --- Bottom face ---
    for t in range(1, n - 1):
        i.append(n)
        j.append(n + t + 1)
        k.append(n + t)

    # --- Side faces (both outward and inward) ---
    for t in range(n):
        t_next = (t + 1) % n
        # outward
        i += [t, t_next, n + t]
        j += [t_next, n + t_next, n + t]
        k += [n + t, n + t_next, t_next]
        # inward (reversed winding order)
        i += [t, n + t, t_next]
        j += [t_next, n + t_next, n + t]
        k += [n + t, t_next, n + t_next]

    # --- Plot ---
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=X, y=Z, z=Y,
                i=i, j=j, k=k,
                color=color,
                flatshading=True,
                opacity=1.0
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    fig.update_layout(height=650,
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ),
        scene_camera=dict(
        eye=dict(x=2, y=2, z=2)  # move camera further away for a zoomed-out view
    ))

    experiment_holder.plotly_chart(fig)
    return fig

option = st.sidebar.selectbox(
    "Inference Quantity",
    (X1_name,X2_name,X3_name),
)

experiment_option = st.sidebar.selectbox(
    "Experiment:",
    ("Experiment A","Experiment B"),
)

constant = 1E9





df_experiment_A = pd.DataFrame({
    "output_flow": [3],
    "thickness": [1.95]

})
df_experiment_B= pd.DataFrame({
    "output_flow": [10],
    "thickness": [1.95]

})

df_experiment_A_std = pd.DataFrame({
    "output_flow": [0.01],
    "thickness": [0.01]

})

df_experiment_B_std = pd.DataFrame({
    "output_flow": [0.01],
    "thickness": [0.01]

})

progress = st.progress(0)
status = st.empty()
column1, column2 = st.columns(2)
experiment_holder = column1.empty()
calib_plot_holder = column2.empty()
fig = update_experiment_plot()
    
if st.sidebar.button("Run Inference"):
# Create polar coordinates




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
        if experiment_option == "Experiment A":
            df_experiment = df_experiment_A
            df_experiment_std = df_experiment_A_std
        else:
            df_experiment = df_experiment_B
            df_experiment_std = df_experiment_B_std
        status.text("Running inference..." )
        calibparams = tl.CalibrateParams(return_summary=False,start_location = "optimized",iterations=100)
        calib_result = emulator.calibrate(df_experiment, df_experiment_std,params=calibparams)
        
    for i in range(0,len(calib_result),1):

        fig3 = px.scatter_matrix(calib_result.iloc[:i],
        dimensions=[X1_name,X2_name,X3_name])
        fig3.update_traces(marker=dict(color = colors[1],opacity=0.3))
        calib_plot_holder.plotly_chart(fig3)
        fig.update_traces(opacity=calib_result["diffusivity"].iloc[i]/10)
        experiment_holder.plotly_chart(fig)
    status.text("Inference Complete!" )
    mean = np.mean(calib_result["diffusivity"])
    std = np.std(calib_result["diffusivity"])
    st.text(f"diffusivity is {mean} pm {std}")

