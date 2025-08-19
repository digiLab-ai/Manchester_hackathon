import galaxy_helpers as gh
from bioblend.galaxy import GalaxyInstance
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# twinLab import
import twinlab as tl

tl.set_api_key("tl_oA4GuDIS9j6MIXEu502IdA")
tl.set_user("cyd@digilab.co.uk")

emulator = tl.Emulator("Tritium_Transport_Emulator")

my_priors = [
    tl.Prior("pressure_inlet", tl.distributions.Uniform(20, 50)),
    tl.Prior("diffusivity", tl.distributions.Uniform(1e-10, 1e-8)),
]

num_points = 20

initial_design = emulator.design(my_priors, num_points=num_points)

api_key = "dc9ac34f13609b805744e65558f52643"
server = "130.88.41.197"

wf_name = "TMAP"

gi = GalaxyInstance(url=server, key=api_key)

file_path = "./tmap.i"

default_params = {
    "solubility": 0.04,
    "Simulation Time": 100.0,
    "temperature": 773.15,
    "V_chamber": 0.0003278,
    "interval_time": 5.0,
    "diffusivity": 1.5e-09,
    "Diameter": 15.0,
    "pressure_inlet": 42000.0,
    "Thickness": 2.0,
}

inputs = initial_design.to_dict()
param_inputs = []
for i in range(num_points):
    param_inputs.append(
        {
            "pressure_inlet": inputs["pressure_inlet"][i],
            "diffusivity": inputs["diffusivity"][i],
        }
    )

print(param_inputs)

for input in param_inputs:
    # Create a copy of the default parameters
    param = default_params.copy()

    # Update the parameters with the input values
    param.update(input)

    param_id = f"{input['pressure_inlet']}_{input['diffusivity']}"

    history = gi.histories.create_history(name=f"TMap Parameter Sweep_{param_id}")

    workflow_inputs = {}
    expected_inputs = gh.get_inputs(
        api_key=api_key, workflow_name=wf_name, uid=param_id
    )

    for input in expected_inputs:
        if input[0] == "dataset":
            # TODO: Implement
            uploads = []
            uploads.append(gi.tools.upload_file(file_path, history["id"]))
            workflow_inputs[str(input[2])] = {
                "src": "hda",
                "id": uploads[-1]["outputs"][0]["id"],
            }
            pass
        elif input[0] == "parameter":
            for name, string in param.items():
                if input[1] == name:
                    workflow_inputs[str(input[2])] = string
        else:
            raise ValueError("Input type not recognised")

    # Pull and launch the workflow
    workflow = gi.workflows.get_workflows(name=wf_name)
    gi.workflows.invoke_workflow(
        workflow_id=workflow[0]["id"], inputs=workflow_inputs, history_id=history["id"]
    )

    # Get the invocation of the above workflow and then wait
    # for it to complete
    invocations = gi.invocations.get_invocations(workflow_id=workflow[0]["id"])
    for invocation in invocations:
        if invocation["history_id"] == history["id"]:
            invocation_id = invocation["id"]
            break
        else:
            continue
    else:
        raise Exception("Invocation lost somewhere...")

    gi.invocations.wait_for_invocation(invocation_id=invocation_id)

    # Find the job created by the invocation and wait for the job
    # to finish (computation done)
    jobs = gi.jobs.get_jobs(invocation_id=invocation_id)
    job_ids = [(job["id"], job["tool_id"]) for job in jobs]
    print(f"Waiting for jobs {job_ids}")
    for id in job_ids:
        print(f"Waiting for job {id[1]} to finish")
        gi.jobs.wait_for_job(job_id=id[0])
        print("Job finished")
    print("All jobs finished")

    # Save all workflow params and outputs via the RO-Crate

    # --- Download CSV output(s) ---
    # List datasets in the history
    datasets = gi.histories.show_history(history["id"], contents=True)

    for ds in datasets:
        if ds["name"] == "CSV Output":
            output_path = f"./outputs/{param_id}_{ds['name']}.csv"
            gi.datasets.download_dataset(
                dataset_id=ds["id"], file_path=output_path, use_default_filename=False
            )
            print(f"Downloaded {ds['name']} to {output_path}")
