# Neater, usable

# Importing libraries
import os
from dotenv import load_dotenv
from uncertainty_engine.client import Client, Environment
from uncertainty_engine.graph import Graph
from uncertainty_engine.nodes.basic import Add
from uncertainty_engine.nodes.base import Node
from uncertainty_engine_types import ResourceID
from uncertainty_engine.nodes.workflow import Workflow

# Third party imports
from typing import Any, Dict, Optional, Union, Iterable, Tuple, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from itertools import product
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from pprint import pprint
from io import BytesIO
from io import StringIO
import requests
import io
from typing import Any, Dict, Optional, Union, Iterable
import json
from io import StringIO
import requests
from uncertainty_engine_types import ResourceID


def train_and_save_model_workflow(
    client,
    project_name: str,
    dataset_name: str,
    input_names: list,
    output_names: list,
    save_model_name: str,
    is_visualise_workflow: bool = False,
    is_print_full_output: bool = False,
) -> dict:
    """
    A workflow that trains a machine learning model.
    Here, we assume all resources have already been uploaded to the cloud.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    :param dataset_name: The name of the dataset.
    :param input_names: The names of the input columns.
    :param output_names: The names of the output columns.
    :param save_model_name: The name to save the trained model as.
    :param is_visualise_workflow: Whether to print the workflow graph.
    :param is_print_full_output: Whether to print the full output of the workflow.
    :return: The response from running the workflow.
    """
    # 1. Create the graph
    graph = Graph()

    # 2. Create relevant nodes, handles, and add to graph:

    # 2.a. Model config node
    model_config = Node(
        node_name="ModelConfig",
        label="Model Config",
    )
    graph.add_node(model_config)  # add to graph
    output_config = model_config.make_handle("config")  # add handle

    # 2.b. Load dataset node
    load_data = Node(
        node_name="LoadDataset",
        label="Load Dataset",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name,
                resource_name=dataset_name,
                resource_type="dataset",
            )
        ),
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(load_data)  # add to graph
    dataset = load_data.make_handle("file")  # add handle

    # 2.b. Filter dataset node for inputs
    input_data = Node(
        node_name="FilterDataset",
        label="Input Dataset",
        columns=input_names,
        dataset=dataset,
    )
    graph.add_node(input_data)  # add to graph
    input_dataset = input_data.make_handle("dataset")  # add handle

    # 2.c. Filter dataset node for outputs
    output_data = Node(
        node_name="FilterDataset",
        label="Output Dataset",
        columns=output_names,
        dataset=dataset,
    )
    graph.add_node(output_data)  # add to graph
    output_dataset = output_data.make_handle("dataset")  # add handle

    # 2.d. Train model node
    train_model = Node(
        node_name="TrainModel",
        label="Train Model",
        config=output_config,
        inputs=input_dataset,
        outputs=output_dataset,
    )
    graph.add_node(train_model)  # add to graph
    output_model = train_model.make_handle("model")  # add handle

    # 2.e. Save model node
    save = Node(
        node_name="Save",
        label="Save",
        data=output_model,
        file_id=save_model_name,
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(save)  # add to graph

    if is_visualise_workflow:
        pprint(graph.nodes)

    workflow = Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        external_input_id=graph.external_input_id,
        requested_output={},
    )

    response = client.run_node(workflow)
    if is_print_full_output:
        pprint(response.model_dump())


def get_project_id(client, project_name: str) -> str:
    """
    Get the project ID from the project name.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    return: The project ID.
    """
    projects = client.projects.list_projects()
    for project in projects:
        if project.name == project_name:
            return project.id
    raise ValueError(f"Project with name {project_name} not found.")


def get_resource_id(
    client, project_name: str, resource_name: str, resource_type: str
) -> str:
    """
    Get the resource ID from the workflow name.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    :param resource_name: The name of the resource.
    :param resource_type: The type of the resource (e.g., 'dataset', 'model', ...).
    return: The resource ID.
    """
    project_id = get_project_id(client, project_name)
    resources = client.resources.list_resources(project_id, resource_type=resource_type)
    for resource in resources:
        if resource.name == resource_name:
            return resource.id
    raise ValueError(f"Resource with name {resource_name} not found.")


def upload_dataset(
    client, project_name, dataset_name, file_path=None, dataset=None, is_replace=True
):
    PROJECT_ID = get_project_id(client, project_name)
    if file_path is None and dataset is not None:

        file_path = f"{dataset_name}.csv"

        df = pd.DataFrame(dataset)
        df.to_csv(file_path, index=False)
    elif file_path is None and dataset is None:
        raise ValueError("Either file_path or dataset must be provided.")
    PROJECT_ID = get_project_id(client, project_name)
    try:
        client.resources.upload(
            project_id=PROJECT_ID,
            name=dataset_name,
            resource_type="dataset",
            file_path=file_path,
        )
    except Exception as e:
        if is_replace:
            client.resources.update(
                project_id=PROJECT_ID,
                resource_id=get_resource_id(
                    client, project_name, dataset_name, resource_type="dataset"
                ),
                resource_type="dataset",
                file_path=file_path,
            )
        else:
            print(f"Error uploading dataset: {e}")
    print(f"Uploaded {dataset_name} to {project_name}")


def wrap_resource_id(resource_id: str):
    """
    Wrap a resource ID in a ResourceID object and return its dictionary representation.
    :param resource_id: The resource ID to wrap.
    :return: The dictionary representation of the ResourceID object."""
    return ResourceID(id=resource_id).model_dump()


def get_resource(client, project_name: str, resource_name: str, resource_type: str):
    """
    Download a resource from the Uncertainty Engine.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    :param dataset_name: The name of the dataset.
    :return: The dataset as a pandas DataFrame.
    """
    response = client.resources.download(
        project_id=get_project_id(client=client, project_name=project_name),
        resource_type=resource_type,
        resource_id=get_resource_id(
            client=client,
            project_name=project_name,
            resource_name=resource_name,
            resource_type=resource_type,
        ),
    )
    decoded = response.decode("utf-8")
    return decoded


def get_data(
    client,
    project_name: str,
    dataset_name: str,
):
    """
    Download a dataset from the Uncertainty Engine and return it as a pandas DataFrame.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    :param dataset_name: The name of the dataset.
    :return: The dataset as a pandas DataFrame.
    """
    decoded = get_resource(
        client=client,
        project_name=project_name,
        resource_name=dataset_name,
        resource_type="dataset",
    )
    df = pd.read_csv(io.StringIO(decoded))
    return df


def get_model(client, project_name: str, model_name: str):

    decoded = get_resource(
        client=client,
        project_name=project_name,
        resource_name=model_name,
        resource_type="model",
    )
    model = json.loads(decoded)
    return model


def get_model_inputs(client, project_name: str, model_name: str) -> Iterable[str]:
    """
    Get the input feature names for a model.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    :param model_name: The name of the model.
    :return: A list of input feature names.
    """
    model = get_model(client=client, project_name=project_name, model_name=model_name)
    input_features = model["metadata"]["inputs"]
    return input_features


def get_node_info(node_name):
    nodes = client.list_nodes()
    nodes_by_id = {node["id"]: node for node in nodes}
    # Print the details of the node
    pprint(nodes_by_id[node_name])


def get_presigned_url(url):
    """
    Get the contents from the presigned url.
    """
    url = url.replace("https://", "http://")
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response


def predict_model_workflow(
    client,
    predict_dataset: Optional[Union[str, pd.DataFrame, Dict[str, Iterable[Any]]]],
    project_name: str,
    model_name: str,
    input_names: Union[list, None] = None,
    is_visualise_workflow: bool = False,
    is_print_full_output: bool = False,
) -> dict:
    """
    A workflow that trains a machine learning model.
    Here, we assume all resources have already been uploaded to the cloud.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    :param dataset_name: The name of the dataset.
    :param input_names: The names of the input columns.
    :param output_names: The names of the output columns.
    :param save_model_name: The name to save the trained model as.
    :param is_visualise_workflow: Whether to print the workflow graph.
    :param is_print_full_output: Whether to print the full output of the workflow.
    :return: The response from running the workflow.
    """
    predict_dataset_name = "_predict_data"
    is_upload_dataset = type(predict_dataset) is not str
    if is_upload_dataset:
        upload_dataset(
            client=client,
            project_name=project_name,
            dataset_name=predict_dataset_name,
            dataset=predict_dataset,
        )
    else:
        predict_dataset_name = predict_dataset

    input_dataset = get_data(
        client=client, project_name=project_name, dataset_name=predict_dataset_name
    )
    if input_names is None:
        # input_names are the input_dataset columns
        input_names = input_dataset.columns.tolist()
    else:
        # Check that the input names exist in the dataset
        missing = set(input_names) - set(input_dataset.columns)
        if missing:
            raise ValueError(
                f"The following input_names are missing from dataset columns: {sorted(missing)}\n"
                f"Available columns: {input_dataset.columns.tolist()}"
            )

    # check the model's expected inputs
    expected_inputs = get_model_inputs(
        client=client,
        project_name=project_name,
        model_name=model_name,
    )
    missing = set(input_names) - set(expected_inputs)
    if missing:
        raise ValueError(
            f"The following input_names are missing from dataset columns: {sorted(missing)}\n"
            f"Available columns: {input_dataset.columns.tolist()}"
        )
    # 1. Create the graph
    graph = Graph()

    # 2. Create relevant nodes, handles, and add to graph:

    # 2.a.

    # 2.a. Load dataset node
    load_data = Node(
        node_name="LoadDataset",
        label="Load Dataset",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name,
                resource_name=predict_dataset_name,
                resource_type="dataset",
            )
        ),
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(load_data)  # add to graph
    dataset = load_data.make_handle("file")  # add handle
    if input_names is not None:
        # 2.b. Filter dataset node for inputs
        input_data = Node(
            node_name="FilterDataset",
            label="Input Dataset",
            columns=input_names,
            dataset=dataset,
        )
        graph.add_node(input_data)  # add to graph
        input_dataset = input_data.make_handle("dataset")  # add handle
    else:
        input_dataset = dataset

    # 2.a. Load model node
    load_model = Node(
        node_name="LoadModel",
        label="Load Model",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name,
                resource_name=model_name,
                resource_type="model",
            )
        ),
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(load_model)  # add to graph
    # 2.d. Predict model node
    predict_model = Node(
        node_name="PredictModel",
        label="Predict Model",
        dataset=input_dataset,
        model=load_model.make_handle("file"),
    )
    graph.add_node(predict_model)  # add to graph

    # 2.e. Display node
    download_predict = Node(
        node_name="Download",
        label="Download Prediction",
        file=predict_model.make_handle("prediction"),
    )
    graph.add_node(download_predict)  # add to graph

    # 2.e. Display node
    download_uncertainty = Node(
        node_name="Download",
        label="Download Uncertainty",
        file=predict_model.make_handle("uncertainty"),
    )
    graph.add_node(download_uncertainty)  # add to graph

    if is_visualise_workflow:
        pprint(graph.nodes)

    workflow = Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        external_input_id=graph.external_input_id,
        requested_output={
            "Predictions": download_predict.make_handle("file").model_dump(),
            "Uncertainty": download_uncertainty.make_handle("file").model_dump(),
        },
    )

    response = client.run_node(workflow)
    if is_print_full_output:
        pprint(response.model_dump())

    # Download the predictions and save as a DataFrame
    predictions_response = get_presigned_url(response.outputs["outputs"]["Predictions"])
    predictions_df = pd.read_csv(
        StringIO(predictions_response.text)
    )  # Save the predictions to a DataFrame

    # Download the uncertainty and save as a DataFrame
    uncertainty_response = get_presigned_url(response.outputs["outputs"]["Uncertainty"])
    uncertainty_df = pd.read_csv(
        StringIO(uncertainty_response.text)
    )  # Save the uncertainty to a DataFrame

    # Clean up if the input dataset was uploaded within this function
    if is_upload_dataset:
        try:
            client.resources.delete_resource(
                project_id=get_project_id(client=client, project_name=project_name),
                resource_id=get_resource_id(
                    client=client,
                    project_name=project_name,
                    resource_name=predict_dataset_name,
                    resource_type="dataset",
                ),
                resource_type="dataset",
            )
        except Exception as e:
            # Non-fatal: log or print if desired; don't mask main exceptions
            print(f"[WARN] Failed to delete temp dataset '{predict_dataset_name}': {e}")

    return predictions_df, uncertainty_df

def create_initial_design(
    client, params_to_emulate,num_points
):
    """
    Get the input feature names for a model.
    :param client: The Uncertainty Engine client.
    :param project_name: The name of the project.
    :param model_name: The name of the model.
    :return: A list of input feature names.
    """
    graph = Graph()
    my_priors = []
    labels = []
    upper_bound = []
    lower_bound = []
    for parameter, bounds in params_to_emulate.items():
        lower_bound.append(bounds[0])
        upper_bound.append(bounds[1])
        labels.append(parameter)


    initialDesign = Node(
        node_name="CreateInitialDesign",
        label="Create Initial Design",
        labels=labels,
        lower_bounds = lower_bound,
        upper_bounds = upper_bound,
        num_points = num_points,
    )

    graph.add_node(initialDesign)  # add to graph

    samples = initialDesign.make_handle("samples")  # add handle


    workflow = Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        external_input_id=graph.external_input_id,
        requested_output={
            "samples": samples.model_dump(),

        },
    )

    response = client.run_node(workflow)
    df = pd.read_csv(StringIO(response.model_dump()['outputs']['outputs']['samples']['csv']))

    return df


def get_presigned_url(url: str) -> requests.Response:
    """
    Fetch the contents of a (pre-signed) URL.

    This function replaces the scheme 'https://' with 'http://' prior to the request,
    mirroring the behavior in the original code (some pre-signed endpoints may require it).

    Parameters
    ----------
    url : str
        The original (likely pre-signed) URL.

    Returns
    -------
    requests.Response
        The HTTP response object. Use `.content` / `.text` to access the payload.

    Raises
    ------
    requests.HTTPError
        If the response indicates an HTTP error status.
    """
    url = url.replace("https://", "http://")
    response = requests.get(url)
    response.raise_for_status()
    return response



def recommend(
    client: Client,
    project_name: str,
    model_name: str,
    number_of_points: int = 1,
    acquisition_function: Literal[
        "ExpectedImprovement",
        "LogExpectedImprovement",
        "PosteriorMean",
        "MonteCarloExpectedImprovement",
        "MonteCarloLogExpectedImprovement",
    ] = "PosteriorMean",
    is_visualise_workflow: bool = False,
    is_print_full_output: bool = False,
    save_workflow_name: Union[str, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and execute a workflow that loads a saved model, proposes up to
    `number_of_points` locations that *maximize the objective* according to the
    chosen acquisition function, and then evaluates the model at those points to
    return predictions and uncertainties.

    The constructed workflow is:
        1) LoadModel
        2) Recommend(acquisition_function, number_of_points)
        3) PredictModel(dataset = recommended_points, model = loaded model)
        4) Download(prediction), Download(uncertainty)

    Parameters
    ----------
    client : Client
        Uncertainty Engine client instance (already authenticated).
    project_name : str
        Name of the project containing the saved model.
    model_name : str
        Name of the saved model resource to load for recommendation and prediction.
    number_of_points : int, default 1
        How many candidate maximizers to propose.
    acquisition_function : {"ExpectedImprovement", "LogExpectedImprovement",
                            "PosteriorMean",
                            "MonteCarloExpectedImprovement", "MonteCarloLogExpectedImprovement"},
        default "PosteriorMean"
        Acquisition used to propose maximizers:
        - **ExpectedImprovement (EI):** Classic improvement over current best (greedy-exploit with exploration).
        - **LogExpectedImprovement:** Log-space EI, numerically stabler for tiny improvements.
        - **PosteriorMean:** Greedy exploitation of the model mean (argmax of mean).
        - **MonteCarloExpectedImprovement (MC-EI):** EI estimated via posterior MC samples (better for batches/multi-modality).
        - **MonteCarloLogExpectedImprovement (MC-log-EI):** Log-space MC EI for numerical stability.


    """
    # 1) Build graph.
    graph = Graph()

    # 2a) Load the trained model.
    load_model = Node(
        node_name="LoadModel",
        label="Load Model",
        file_id=wrap_resource_id(
            get_resource_id(
                client=client,
                project_name=project_name,
                resource_name=model_name,
                resource_type="model",
            )
        ),
        project_id=get_project_id(client=client, project_name=project_name),
    )
    graph.add_node(load_model)

    # 2d) Predict with uncertainty.
    recommend_model = Node(
        node_name="Recommend",
        label="Recommend",
        model=load_model.make_handle("file"),
        acquisition_function=acquisition_function,
        number_of_points=number_of_points,
    )
    graph.add_node(recommend_model)

    download_recommend = Node(
        node_name="Download",
        label="Download Recommend",
        file=recommend_model.make_handle("recommended_points"),
    )
    graph.add_node(download_recommend)


    # Finalize workflow payload (explicitly request both artifacts).
    workflow = Workflow(
        graph=graph.nodes,
        inputs=graph.external_input,
        external_input_id=graph.external_input_id,
        requested_output={
            "sample": download_recommend.make_handle("file").model_dump(),
        
        },
    )
    # Execute inference.
    response = client.run_node(workflow)
    rec_response = get_presigned_url(response.outputs["outputs"]["sample"])
    rec_df = pd.read_csv(StringIO(rec_response.text))

    
    return rec_df

