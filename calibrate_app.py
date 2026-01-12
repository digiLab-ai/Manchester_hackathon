import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time
import os
import plotly.graph_objects as go
import engine_helpers as engine
from uncertainty_engine.client import Client, Environment

os.environ["UE_USERNAME"] = "cyd.cowley@digilab.ai"
os.environ["UE_PASSWORD"] = ""

client = Client(env="dev")
client.authenticate()

import plotly.express as px

colors = ["#16425B", "#16D5C2", "#EBF38B"]


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


# project IDs
projects_dict = {proj.name: proj.id for proj in client.projects.list_projects()}
PROJECT_NAME = "UoM_LIBRTI"
obs_file = "Data//observation.csv"
noise_file = "Data//uncertainty.csv"
input_columns = ["diffusivity", "solubility", "thickness"]


lower_bounds = [0.1,0.02,1.99]
upper_bounds = [10,0.08,2.0]
obs_dataset_name = "observations"
noise_dataset_name = "uncertainty"


def score(model_name, bounds):

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
            resource_id=engine.get_resource_id(
                client, PROJECT_NAME, dataset_id_test, resource_type="dataset"
            ),
            resource_type="dataset",
            file_path=validation_file,
        )
    pred_grid, std_grid = engine.predict_model_workflow(
        client=client,
        predict_dataset=dataset_id_test,
        project_name=PROJECT_NAME,
        model_name=model_name,
        is_print_full_output=False,
        input_names=input_columns,
    )

    # return average uncertainty of predictions
    return np.mean(np.array(std_grid).flatten()) / np.mean(
        np.array(pred_grid).flatten()
    )

def update_experiment_plot(title="GDPS sample, unkown diffusivity"):

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
    fig.update_layout(title=title)
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



experiment_option = st.sidebar.selectbox(
    "Experiment:",
    ("Experiment A","Experiment B"),
)





df_experiment_A = pd.DataFrame({
    "output_flow": [3.0],
    # "thickness": [1.95]

})
df_experiment_B= pd.DataFrame({
    "output_flow": [13.0],
    # "thickness": [1.95]

})

df_experiment_A_std = pd.DataFrame({
    "output_flow": [0.01],
    # "thickness": [0.01]

})

df_experiment_B_std = pd.DataFrame({
    "output_flow": [0.01],
    # "thickness": [0.01]

})
uncertainty_cutoff = 0.3
progress = st.progress(0)
status = st.empty()
column1, column2 = st.columns(2)
experiment_holder = column1.empty()
calib_plot_holder = column2.empty()
fig = update_experiment_plot()
    
if st.sidebar.button("Run Inference"):
    if experiment_option =="Experiment A":
        df_experiment_A.to_csv(obs_file, index=False)
        df_experiment_A_std.to_csv(noise_file, index=False)
    elif experiment_option =="Experiment B":
        df_experiment_B.to_csv(obs_file, index=False)
        df_experiment_B_std.to_csv(noise_file, index=False)

    try:
        client.resources.upload(
            project_id=projects_dict[PROJECT_NAME],
            name=obs_dataset_name,
            resource_type="dataset",
            file_path=obs_file,
        )
    except:
        client.resources.update(
            project_id=projects_dict[PROJECT_NAME],
            resource_id=engine.get_resource_id(
                client, PROJECT_NAME, obs_dataset_name, resource_type="dataset"
            ),
            resource_type="dataset",
            file_path=obs_file,
        )
    try:
        client.resources.upload(
            project_id=projects_dict[PROJECT_NAME],
            name=noise_dataset_name,
            resource_type="dataset",
            file_path=noise_file,
        )
    except:
        client.resources.update(
            project_id=projects_dict[PROJECT_NAME],
            resource_id=engine.get_resource_id(
                client, PROJECT_NAME, noise_dataset_name, resource_type="dataset"
            ),
            resource_type="dataset",
            file_path=noise_file,
        )
    df_experiment = 0
    df_experiment_std = 0
    MODEL_NAME = "UoM_TMAP_model"

    try:
        status.text("checking for existing emulator...")

        __ = engine.get_model_inputs(
            client=client,
            project_name=PROJECT_NAME,
            model_name=MODEL_NAME,
        )

        # uncertainty_score = score(MODEL_NAME, bounds)

        emulator_exists = 1


        # if uncertainty_score > uncertainty_cutoff:
        #     status.text("emulator exists but is not valid...")

        # else:
        #     # in the case that an emulator isn't found

        #     status.text(
        #         f"emulator already exists with uncertainty of {100*round(uncertainty_score,3)} %"
        #     )
        #     emulator_performs = 1
    except:
        emulator_exists = 0

    calib_result = 0
    if emulator_exists:
        if experiment_option == "Experiment A":
            df_experiment = df_experiment_A
            df_experiment_std = df_experiment_A_std
        else:
            df_experiment = df_experiment_B
            df_experiment_std = df_experiment_B_std
        status.text("Running inference..." )
        project_name = "UoM_LIBRTI"



        calib_result = engine.infer_model_workflow(client=client,
                            project_name=project_name,
                            model_name=MODEL_NAME,
                            obs_dataset_name=obs_dataset_name,
                            noise_dataset_name=noise_dataset_name,
                            lower_bounds=lower_bounds,
                            upper_bounds=upper_bounds,
                            nsamples=1000)

        colors_diff = []

        fig3 = px.scatter(x=calib_result[:,0],y=calib_result[:,1])
        fig3.update_traces(marker=dict(color = colors[1],opacity=0.3))
        fig3.update_layout(height=650)
        fig3.update_layout(
    xaxis_title="Diffusivity (1E-10 m²/s)",
    yaxis_title="solubility (au)"
)
        calib_plot_holder.plotly_chart(fig3)

        time.sleep(1)
        status.text("Inference Complete!" )
    else:
        status.text("No emulator")

    mean = np.mean(calib_result[:,0])
    std = np.std(calib_result[:,0])
    update_experiment_plot(title=f"GDPS sample, diffusivity is {round(mean,3)} ± {round(std,3)}")

