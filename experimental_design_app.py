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

colors = ["#16425B", "#16D5C2", "#EBF38B"]
# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Emulator Training", layout="wide")

st.title("AI Experiment Design")


# -------------------------------
# Sidebar - Simulation Settings
# -------------------------------
st.sidebar.image("digiLab_logo.png", use_container_width=True)





# model type
st.sidebar.header("Settings")

simulator_type = st.sidebar.selectbox(
    "Experiment",
    ("1_55mm_ni","1_55mm_T"),
)

# project IDs
projects_dict = {proj.name: proj.id for proj in client.projects.list_projects()}
PROJECT_NAME = "UoM_LIBRTI"

# dataset name for data uploaded to the engine
dataset_name = "experiment_design_0"
dataset_name_pred = "experiment_design_pred_0"

# dataset name for simulation outputs
filepathoriginal = "Data//1_55mm_ni.xlsx"
filepathFinal = "Data//1_55mm_ni.csv"
filepathPred = "Data//1_55mm_ni_pred.csv"

output_columnOriginal = "Diffusivity (m²/s)"
output_columnFinal = "Diffusivity (1E-10 m²/s)"
input_column = "Temperature (°C)"

MODEL_NAME = f"UoM_{simulator_type}_model"

predictions = 0






# to adjust diffusivity
constant = 1e10
progress = st.progress(0)
status = st.empty()
status2 = st.empty()

dforiginal = pd.read_excel(filepathoriginal,sheet_name=0)
dffinal = dforiginal[[input_column,output_columnOriginal]]
dffinal = dffinal.rename(columns={output_columnOriginal: output_columnFinal})
dffinal[output_columnFinal] = dffinal[output_columnFinal]*constant
dffinal = dffinal.groupby(input_column, as_index=False)[output_columnFinal].mean()
dffinal = dffinal[dffinal[input_column] != 600]
dffinal.to_csv(filepathFinal)

df_pred = pd.DataFrame({
    input_column:np.linspace(min(dffinal[input_column]),max(dffinal[input_column]),100)
})
df_pred.to_csv(filepathPred)
if st.sidebar.button("Recommend new experiment"):

    try:
        client.resources.upload(
            project_id=projects_dict[PROJECT_NAME],
            name=dataset_name,
            resource_type="dataset",
            file_path=filepathFinal,
        )
    except:
        client.resources.update(
            project_id=projects_dict[PROJECT_NAME],
            resource_id=engine.get_resource_id(
                client, PROJECT_NAME, dataset_name, resource_type="dataset"
            ),
            resource_type="dataset",
            file_path=filepathFinal,
        )
    try:
        client.resources.upload(
            project_id=projects_dict[PROJECT_NAME],
            name=dataset_name_pred,
            resource_type="dataset",
            file_path=filepathPred,
        )
    except:
        client.resources.update(
            project_id=projects_dict[PROJECT_NAME],
            resource_id=engine.get_resource_id(
                client, PROJECT_NAME, dataset_name_pred, resource_type="dataset"
            ),
            resource_type="dataset",
            file_path=filepathPred,
        )
    uncertainty_score = 1
    UQ_score_history = []
    batches = []
    i = 0
    emulator_exists = 0
    emulator_performs = 0
    plot_holder = st.empty()


    df =pd.read_csv(filepathFinal)

    fig = px.scatter(
    df,
    x=input_column,
    y=output_columnFinal,
    
)
    fig.update_traces(name="experiment data", showlegend=True,    marker=dict(color="black"))



    fig.update_layout(height=550)
    plot_holder.plotly_chart(fig)
    try:
        status.text("checking for existing emulator...")

        __ = engine.get_model_inputs(
            client=client,
            project_name=PROJECT_NAME,
            model_name=MODEL_NAME,
        )

        emulator_exists = 1
        status.text(
            f"emulator already exists"
        )

    except:
        emulator_exists = 0
        status.text(
            f"emulator does not exist"
        )
    if not emulator_exists:


        status2.text(f"model training")
        output = engine.train_and_save_model_workflow(
            client,
            project_name=PROJECT_NAME,
            dataset_name=dataset_name,
            input_names=[input_column],
            output_names=[output_columnFinal],
            save_model_name=MODEL_NAME,
            is_print_full_output=True,
        )
        status2.text(f"training complete")


    pred_grid, std_grid = engine.predict_model_workflow(
        client=client,
        predict_dataset=dataset_name_pred,
        project_name=PROJECT_NAME,
        model_name=MODEL_NAME,
        is_print_full_output=False,
        input_names=[input_column],
    )


    fig = px.line( x=df_pred[input_column], y=pred_grid[output_columnFinal]-std_grid[output_columnFinal])
    fig.update_traces(
    line=dict(color=colors[1])
)
    fig.add_scatter(
        x=df_pred[input_column],
        y=pred_grid[output_columnFinal]+std_grid[output_columnFinal],
        fill="tonexty",
        fillcolor=colors[1],
        line=dict(color=colors[1]),
        name="ML model"
    )
    fig.add_scatter(
    x=df[input_column],
    y=df[output_columnFinal],
    name="experiment data",
        mode="markers",
    marker=dict(color="black")
)
    fig.update_layout(
    xaxis_title=input_column,
    yaxis_title=output_columnFinal
)
    plot_holder.plotly_chart(fig)
    

    result = engine.recommend(client=client,
            project_name=PROJECT_NAME,
            model_name=MODEL_NAME,
            number_of_points=1,
            acquisition_function="MonteCarloNegativeIntegratedPosteriorVariance")
    fig.add_vline(
    x=result[input_column][0],
    line_dash="dash",
    line_color="red"
)
    
    status.text(f"new experiment recommended at T={round(result[input_column][0])} °C")
    fig.update_layout(
    xaxis_title=input_column,
    yaxis_title=output_columnFinal
)
    plot_holder.plotly_chart(fig)
    
