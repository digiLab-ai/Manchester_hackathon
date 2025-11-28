import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time
import os

import engine_helpers as engine
from uncertainty_engine.client import Client, Environment
os.environ["UE_USERNAME"] = "cyd.cowley@digilab.ai"
os.environ["UE_PASSWORD"] = ""

client = Client()
client.authenticate()

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

# model type
simulator_type = st.sidebar.selectbox(
    "Simulation Tool",
    ("TMAP", "UOM_Microstructure"),
)

# project IDs
projects_dict = {proj.name: proj.id for proj in client.projects.list_projects()}
PROJECT_NAME = "UoM_LIBRTI"

# dataset name for data uploaded to the engine
dataset_name = "active_learning_0"

# dataset name for simulation outputs
filepath = "Data//tmap-active-loop_training_data.csv"

output_column = "output_flow"
input_columns = ["diffusivity","solubility","thickness"]

MODEL_NAME = f'UoM_{simulator_type}_model'


simulation_output_file = "Data//active_learning_output.csv"
validation_file = "Data//validation.csv"



bounds = pd.DataFrame({
    X1_name: (thickness_min,thickness_max),
    X2_name: (solubility_min,solubility_max),
    X3_name: (diffusivity_min,diffusivity_max),
})

npoints = 4
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


})

pred_data.to_csv(validation_file)

# advanced model training settings
st.sidebar.header("Advanced Settings")

uncertainty_cutoff = st.sidebar.slider("Model quality cutoff (Average Uncertainty %)", 0.0, 100., 10.0, step=0.01)/100
batch_size = st.sidebar.slider("Active learning batch size", 1, 10, 5, step=1)

# to adjust diffusivity
constant = 1E9


# replace with galaxy TMAP workflow
def run_simulator(i):
    # if simulator_type=="TMAP":
    # else:
    
    df = pd.read_csv("Data//tmap-active-loop_training_data.csv")
    df_input = pd.DataFrame({
    "solubility": df["solubility"][:batch_size*(i+1)],
    "diffusivity":constant*(df["diffusivity"][:batch_size*(i+1)]),
    "thickness": df["thickness"][:batch_size*(i+1)],
    "output_flow":df["output_flow"][:batch_size*(i+1)]
    })

    df_input.to_csv(simulation_output_file)





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


def score(model_name,bounds):

    # Upload the dataset
    dataset_id_test = "test_data"
    try:
        client.resources.upload(
            project_id=projects_dict[PROJECT_NAME],
            name=dataset_id_test,
            resource_type="dataset",
            file_path=validation_file,
        )
    except:
        client.resources.update(
            project_id=projects_dict[PROJECT_NAME],
            resource_id=engine.get_resource_id(client, PROJECT_NAME, dataset_id_test, resource_type="dataset"),
            resource_type="dataset",
            file_path=validation_file,
        )
    pred_grid, std_grid = engine.predict_model_workflow(
        client=client,
        predict_dataset=dataset_id_test,
        project_name=PROJECT_NAME,
        model_name=model_name,
        is_print_full_output=False,
        input_names = input_columns
    )

    # return average uncertainty of predictions
    return np.mean(np.array(std_grid).flatten())/np.mean(np.array(pred_grid).flatten())
    

    
# initialize progress bar and plot containers
progress = st.progress(0)
status = st.empty()
status2 = st.empty()
column1, column2 = st.columns(2)


if st.sidebar.button("Train Model"):

    uncertainty_score = 1
    UQ_score_history = []
    batches = []
    i = 0
    emulator_exists = 0
    emulator_performs = 0
    training_plot_holder = column1.empty()
    score_plot_holder = column2.empty()

    df_input = {
        "diffusivity": [],
        "solubility": [],
        "thickness": [],
        "output_flow": []
    }
    try:
        status.text("checking for existing emulator..." )

        __ = engine.get_model_inputs(
        client=client,
        project_name=PROJECT_NAME,
        model_name=MODEL_NAME,
        )

        uncertainty_score = score(MODEL_NAME,bounds)


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



            status2.text(f"Running {simulator_type} simulations")
            if not emulator_exists:
                # replace with initial batch
                run_simulator(i)
                emulator_exists = 1
            else:
                # replace with active learning
                run_simulator(i)
            

            # Intialise a Dataset object
            

            status2.text(f"Dataset uploading")

            # Upload the dataset
            try:
                client.resources.upload(
                    project_id=projects_dict[PROJECT_NAME],
                    name=dataset_name,
                    resource_type="dataset",
                    file_path=simulation_output_file,
                )
            except:
                client.resources.update(
                    project_id=projects_dict[PROJECT_NAME],
                    resource_id=engine.get_resource_id(client, PROJECT_NAME, dataset_name, resource_type="dataset"),
                    resource_type="dataset",
                    file_path=simulation_output_file,
                )
            df_input = engine.get_data(
            client=client,
            project_name=PROJECT_NAME,
            dataset_name=dataset_name
            )

            update_3dplot(df_input)



            status2.text(f"Emulator training...")
            # Train the emulator using the train method
            engine.train_and_save_model_workflow(client, 
                   project_name=PROJECT_NAME,
                   dataset_name=dataset_name, 
                   input_names=input_columns,
                   output_names=[output_column],
                   save_model_name=MODEL_NAME,
                   is_print_full_output=True
                   )
            
            uncertainty_score = score(MODEL_NAME,bounds)

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

