import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time
import os
# from uncertainty_engine.client import Client, Environment
# from uncertainty_engine.graph import Graph
# from uncertainty_engine.nodes.basic import Add
# from uncertainty_engine.nodes.base import Node
# from uncertainty_engine_types import ResourceID
# from uncertainty_engine.nodes.workflow import Workflow
import twinlab as tl

# tl.set_api_key("")
import plotly.express as px
colors = ["#16425B","#16D5C2","#EBF38B"]
# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Emulator Training", layout="wide")

st.title("Training Dashboard")

# -------------------------------
# Sidebar - Simulation Settings
# -------------------------------
st.sidebar.image("digiLab_logo.png",  use_container_width=True)
st.sidebar.header("Model Parameter Range")


# input specification
X1_name = "thickness"
X1_units = "mm"
X2_name = "solubility"
X2_units = "au"
X3_name = "diffusivity"
X3_units = "1E19"
output_column = "output_flow"

# Data generation parameters
diffusivity_min, diffusivity_max = st.sidebar.slider(f"{X3_name} input range [{X3_units}]", 0.1, 10.0, (0.1, 10.0))
solubility_min, solubility_max = st.sidebar.slider(f"{X2_name} input range [{X2_units}]", 0.01, 0.1, (0.01, 0.1))
thickness_min, thickness_max = st.sidebar.slider(f"{X1_name} input range [{X1_units}]", 1., 3., (1.9, 2.0),step=0.1)

bounds = pd.DataFrame({
    X1_name: (thickness_min,thickness_max),
    X2_name: (solubility_min,solubility_max),
    X3_name: (diffusivity_min,diffusivity_max),
})

# advanced model training settings
st.sidebar.header("Advanced Settings")

uncertainty_cutoff = st.sidebar.slider("Model quality cutoff (Average Uncertainty %)", 0.0, 100., 10.0, step=0.01)/100
batch_size = st.sidebar.slider("Active learning batch size", 4, 20, 4, step=1)

# to adjust diffusivity
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





def update_3dplot(df_input):
    # plot new points
    fig = px.scatter_3d(
    df_input,
    x="thickness", y="solubility", z="diffusivity",color="output_flow",
    color_continuous_scale=colors, 
    )
    # update layour and bounds
    fig.update_traces(marker=dict( size=5))
    fig.update_layout(height=550,
                scene=dict(
                    xaxis=dict(range=bounds[X1_name]),
                    yaxis=dict(range=bounds[X2_name]),
                    zaxis=dict(range=bounds[X3_name]),))

    training_plot_holder.plotly_chart(fig)


def score(emulator,bounds):
    npoints = 3
    # create a meshgrid of points within bounds
    X, Y, Z = np.meshgrid(np.linspace(bounds[X1_name][0],bounds[X1_name][1],num=npoints),
                        np.linspace(bounds[X2_name][0],bounds[X2_name][1],num=npoints),
                        np.linspace(bounds[X3_name][0],bounds[X3_name][1],num=npoints),
                        )

    # prediction within specified bounds
    pred_data = pd.DataFrame({
        X1_name: X.flatten(),
        X2_name: Y.flatten(),
        X3_name: Z.flatten(),
        "output_flow": Z.flatten()*0,


    })
    pred_grid, std_grid = emulator.predict(pred_data)
    # return average uncertainty of predictions
    return np.mean(np.array(std_grid).flatten())
    

    
# initialize progress bar and plot containers
progress = st.progress(0)
status = st.empty()
status2 = st.empty()
column1, column2 = st.columns(2)

# global variables
emulator_id = "Manchester_GDPS_Emulator"
dataset_id = "UoM_dataset"

if st.sidebar.button("Train Model"):

    uncertainty_score = 1
    UQ_score_history = []
    batches = []
    i = 0
    emulator_exists = 0
    emulator_performs = 0
    training_plot_holder = column1.empty()
    score_plot_holder = column2.empty()
    emulator = tl.Emulator(id=emulator_id)

    df_input = {
        "diffusivity": [],
        "solubility": [],
        "thickness": [],
        "output_flow": []
    }
    try:
        status.text("checking for existing emulator..." )
        df_input = emulator.view_train_data()
        scoreparams = tl.ScoreParams(metric="R2")
        uncertainty_score = score(emulator,bounds)/np.mean(df_input[output_column])


        emulator_exists = 1

        batches.append(i)
        i = i+1
        UQ_score_history.append(uncertainty_score)
        if uncertainty_score>uncertainty_cutoff:
            status.text("emulator exists but is not valid..." )    


        else:
            # in the case that an emulator isn't found

            status.text(f"emulator already exists with uncertainty of {100*round(uncertainty_score,3)} %" )
            emulator_performs = 1
    except:
        emulator_exists = 0
    time.sleep(1)

    update_3dplot(df_input)
    df_score = pd.DataFrame({
        "batches": batches,
        "UQ_score_history": UQ_score_history
    })

    if not emulator_performs:
        fig2 = px.line(
        df_score,
        x="batches", y="UQ_score_history",
        )
        fig2.update_traces(marker=dict( size=5))
        fig2.update_layout(height=550)
        score_plot_holder.plotly_chart(fig2)
        while uncertainty_score >uncertainty_cutoff:

            status.text(f"active learning batch {i}")



            emulator = tl.Emulator(id=emulator_id)
            if not emulator_exists:
                # replace with initial batch
                df_input = run_simulator(i)
                emulator_exists = 1
            else:
                # replace with active learning
                df_input = run_simulator(i)
            

            # Intialise a Dataset object
            
            dataset = tl.Dataset(id=dataset_id)
            status2.text(f"Dataset uploading")

            # Upload the dataset
            dataset.upload(df_input, verbose=True)
            update_3dplot(df_input)


            params = tl.TrainParams(
                train_test_ratio=0.7,
                estimator="gaussian_process_regression",

            )
            status2.text(f"Emulator training...")
            # Train the emulator using the train method
            emulator.train(
                dataset=dataset,
                inputs=["diffusivity","solubility","thickness"],
                outputs=[output_column],
                params=params,
                verbose=True,
    )
            uncertainty_score = score(emulator,bounds)/np.mean(df_input[output_column])

            batches.append(i)
            UQ_score_history.append(uncertainty_score)
            df_score = pd.DataFrame({
                "batches": batches,
                "UQ_score_history": UQ_score_history
            })



            fig2 = px.line(
            df_score,
            x="batches", y="UQ_score_history",
            )
            fig2.update_traces(marker=dict( size=5))
            fig2.update_layout(height=550)
            score_plot_holder.plotly_chart(fig2)

            progress.progress(min(int(uncertainty_cutoff/uncertainty_score*100),100))

            i = i+1

            time.sleep(1)  # simulate work
        status.text(f"uncertainty is {100*round(uncertainty_score,3)} %, training complete!")
    # st.session_state["data"] = generate_data(n_samples, x_min, x_max, noise)

if st.sidebar.button("Delete Model"):
    emulator_id = "Manchester_GDPS_Emulator"
    emulator = tl.Emulator(id=emulator_id)

    try:
        emulator.delete()
        status.text("emulator deleted")

    except:
        status.text("no emulator to delete")
