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

tl.set_api_key("tl_zATKGbFWx34DVRj9wIUryw")
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

X1_name = "thickness"
X1_units = "mm"
X2_name = "solubility"
X2_units = "au"
X3_name = "diffusivity"
X3_units = "1E19"
# Data generation parameters


diffusivity_min, diffusivity_max = st.sidebar.slider(f"{X3_name} input range [{X3_units}]", 0.1, 10.0, (0.1, 10.0))
solubility_min, solubility_max = st.sidebar.slider(f"{X2_name} input range [{X2_units}]", 0.01, 0.1, (0.01, 0.1))
thickness_min, thickness_max = st.sidebar.slider(f"{X1_name} input range [{X1_units}]", 1., 3., (1.9, 2.0),step=0.1)

st.sidebar.header("Advanced Settings")

r_squared_cutoff = st.sidebar.slider("Model quality cutoff (R squared)", 0., 1., 0.98, step=0.01)
batch_size = st.sidebar.slider("Active learning batch size", 4, 20, 4, step=1)

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

if st.sidebar.button("Train Model"):
    # initialize low R2
    R2 = 0
    R2_history = []
    batches = []
    i = 0
    training_plot_holder = column1.empty()
    score_plot_holder = column2.empty()
    emulator_id = "Manchester_GDPS_Emulator"
    emulator = tl.Emulator(id=emulator_id)
    try:
        status.text("checking for existing emulator..." )
        emulator.view_train_data()
        scoreparams = tl.ScoreParams(metric="R2")
        R2 = np.array(emulator.score(params=scoreparams)).flatten()[0]
        if R2<r_squared_cutoff:
            # activelearning
            status.text("emulator exists but is not valid..." )       
        else:

            status.text(f"emulator already exists with R squared of {R2}" )
    except:
        
        while R2 <r_squared_cutoff:
            status.text(f"active learning batch {i}")
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim([thickness_min,thickness_max])
            ax.set_ylim([solubility_min,solubility_max])
            ax.set_zlim([diffusivity_min,diffusivity_max])
            ax.set_xlabel(f"{X1_name} [{X1_units}]")
            ax.set_ylabel(f"{X2_name} [{X2_units}]")
            ax.set_zlabel(f"{X3_name} [{X3_units}]")
            ax.grid(False)
            fig.tight_layout()
            fig2, ax2 = plt.subplots(1,1,figsize=(6, 5))
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.set_xlabel("batch number")
            ax2.set_ylabel(r"$R^{2}$")
            fig2.tight_layout()
            if i==0:
                training_plot_holder.pyplot(fig)
                score_plot_holder.pyplot(fig2)
            emulator_id = "Manchester_GDPS_Emulator"
            emulator = tl.Emulator(id=emulator_id)

            df_input = run_simulator(i)
            
            dataset_id = "UoM_dataset"
            # Intialise a Dataset object
            
            dataset = tl.Dataset(id=dataset_id)
            status.text(f"Dataset uploading")

            # Upload the dataset
            dataset.upload(df_input, verbose=True)
            ax.scatter(df_input["thickness"], df_input["solubility"], df_input["diffusivity"], color="black", cmap='viridis')
            fig = px.scatter_3d(
            df_input,
            x="thickness", y="solubility", z="diffusivity",color="output_flow",
            color_continuous_scale=colors, 
        )
            fig.update_traces(marker=dict( size=5))
            training_plot_holder.plotly_chart(fig)

            output_columns = ["output_flow"]
            params = tl.TrainParams(
                train_test_ratio=0.7,
                estimator="gaussian_process_regression",

            )
            status.text(f"Emulator training...")
            # Train the emulator using the train method
            emulator.train(
                dataset=dataset,
                inputs=["diffusivity","solubility","thickness"],
                outputs=output_columns,
                params=params,
                verbose=True,
    )
            scoreparams = tl.ScoreParams(metric="R2")
            R2 = np.array(emulator.score(params=scoreparams)).flatten()[0]
            if R2<0:
                R2 = 0
            batches.append(i)
            R2_history.append(R2)

            ax2.plot(batches,R2_history,marker="o",color=colors[1])
            ax2.set_xlim([0,10])
            ax2.set_ylim([0,1])
            score_plot_holder.pyplot(fig2)



            progress.progress(int(R2*100))

            i = i+1

            # status.text(f"Running step {i + 1}/100")
            time.sleep(1)  # simulate work
        progress.progress(100)
        status.text(f"R squared is {R2}, training complete!")
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
