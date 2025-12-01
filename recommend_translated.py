
import pandas as pd
import os
from dotenv import load_dotenv
from uncertainty_engine.client import Client, Environment
from uncertainty_engine.graph import Graph
from uncertainty_engine.nodes.basic import Add
from uncertainty_engine.nodes.base import Node
from uncertainty_engine_types import ResourceID
from uncertainty_engine.nodes.workflow import Workflow
from engine_helpers import create_initial_design, recommend,get_presigned_url
from pprint import pprint
from io import StringIO

os.environ["UE_USERNAME"] = "cyd.cowley@digilab.ai"
os.environ["UE_PASSWORD"] = ""

num_points = 20

client = Client()
client.authenticate()
client.list_nodes() 
params = {
    "diffusivity": [0.,1.],
    "solubility": [2.,4.]
}

nodes = client.list_nodes()

result = create_initial_design(client,params,num_points)
result = recommend(client=client,project_name="UoM_LIBRTI",model_name="UoM_TMAP_model",number_of_points=4,acquisition_function="MonteCarloNegativeIntegratedPosteriorVariance")
