from bioblend.galaxy import GalaxyInstance
import galaxy_helpers as gh

import numpy as np
import pandas as pd

import twinlab as tl

TWINLAB_API_KEY = ''
TWINLAB_API_USER = ''

GALAXY_API_KEY = ''
GALAXY_SERVER_ADDR = ''

GALAXY_WORKFLOW_NAME = 'TMAP'

DEFAULT_PARAMS = {
    'solubility': 0.04,
    'Simulation Time': 100.0,
    'temperature': 773.15,
    'V_chamber': 0.0003278,
    'interval_time': 5.0,
    'diffusivity': 1.5e-09,
    'Diameter': 15.0,
    'pressure_inlet': 42000.0,
    'Thickness': 2.0
}

# Format of 'parameter name': [min_bound, max_bound]
TWINLAB_PARAMS_TO_EMULATE = {
    'pressure_inlet': (20.0, 50.0),
    'diffusivity': (1e-10, 1e-8)
}

WORKFLOW_OUTPUT_PARAMS = [
    'output_flow'
]

FILE_PATH = './tmap.i'


def create_training_points(num_points):
    """
    Create training points for the specified parameters.

    Usage:
        create_training_points(num_points=100)

    Args:
        num_points: The number of training points to generate.

    Returns:
        A DataFrame containing the generated training points.
    """
    my_priors = []
    for parameter, bounds in TWINLAB_PARAMS_TO_EMULATE.items():
        my_priors.append(tl.Prior(parameter, tl.distributions.Uniform(bounds[0], bounds[1])))

    emulator = tl.Emulator('expt design emulator')
    training_points = emulator.design(my_priors, n_samples=num_points)
    return training_points


def run_galaxy_workflows(gi, dataset, num_points):
    """
    Run Galaxy workflows on the given dataset.

    Args:
        gi: GalaxyInstance object.
        dataset: The dataset to process.
        num_points: The number of points to process.

    Returns:
        The output of the workflow.
    """
    param_inputs = []
    inputs = dataset.to_dict()
    for i in range(num_points):
        param = DEFAULT_PARAMS.copy()
        input = {key: inputs[key][i] for key in inputs if key in DEFAULT_PARAMS}
        param.update(input)
        param_inputs.append(param)

    columns = list(param_inputs[0].keys())
    for item in WORKFLOW_OUTPUT_PARAMS:
        columns.append(item)
    workflow_output = pd.DataFrame(columns=columns)

    for input in param_inputs:
        param_id = f"{input['pressure_inlet']}_{input['diffusivity']}"

        history = gi.histories.create_history(name=f"TMap Parameter Sweep_{param_id}")

        workflow_inputs = {}
        expected_inputs = gh.get_inputs(
            api_key=GALAXY_API_KEY,
            workflow_name=GALAXY_WORKFLOW_NAME,
            uid=param_id
        )

        for input in expected_inputs:
            if input[0] == "dataset":
                # TODO: Implement
                uploads = []
                uploads.append(
                    gi.tools.upload_file(FILE_PATH, history['id'])
                )
                workflow_inputs[str(input[2])] = {
                    'src': 'hda',
                    'id': uploads[-1]['outputs'][0]['id']
                }
                pass
            elif input[0] == "parameter":
                for name, string in param.items():
                    if input[1] == name:
                        workflow_inputs[str(input[2])] = string
            else:
                raise ValueError("Input type not recognised")

        # Pull and launch the workflow
        workflow = gi.workflows.get_workflows(name=GALAXY_WORKFLOW_NAME)
        gi.workflows.invoke_workflow(
            workflow_id=workflow[0]['id'],
            inputs=workflow_inputs,
            history_id=history['id']
        )

        # Get the invocation of the above workflow and then wait
        # for it to complete
        invocations = gi.invocations.get_invocations(
            workflow_id=workflow[0]['id']
        )
        for invocation in invocations:
            if invocation['history_id'] == history['id']:
                invocation_id = invocation['id']
                break
            else:
                continue
        else:
            raise Exception('Invocation lost somewhere...')

        gi.invocations.wait_for_invocation(invocation_id=invocation_id)

        # Find the job created by the invocation and wait for the job
        # to finish (computation done)
        jobs = gi.jobs.get_jobs(invocation_id=invocation_id)
        job_ids = [(job['id'], job['tool_id']) for job in jobs]
        print(f'Waiting for jobs {job_ids}')
        for id in job_ids:
            print(f'Waiting for job {id[1]} to finish')
            gi.jobs.wait_for_job(job_id=id[0])
            print('Job finished')
        print('All jobs finished')

        # Save all workflow params and outputs via the RO-Crate

        # --- Download CSV output(s) ---
        # List datasets in the history
        datasets = gi.histories.show_history(history['id'], contents=True)
        print(datasets)

        for ds in datasets:
            if ds['name'] == 'CSV Output':
                output_path = f"./outputs/{param_id}_{ds['name']}.csv"
                gi.datasets.download_dataset(
                    dataset_id=ds['id'],
                    file_path=output_path,
                    use_default_filename=False
                )
                print(f"Downloaded {ds['name']} to {output_path}")

        # Get params from workflow
        rocrate_filepath = f"./{param_id}.zip"
        gi.invocations.get_invocation_rocrate_zip(
            invocation_id=invocation_id,
            file_path=rocrate_filepath
        )

        # Get parameters from galaxy
        parameters = gh.get_parameters_from_invocation(
            api_key=GALAXY_API_KEY,
            invocation_id=invocation_id,
            workflow_name=GALAXY_WORKFLOW_NAME,
            uid=param_id
        )

# TODO: test
        if len(parameters) > 0:
            print(f"Parameters: {parameters}")
            output_params = parameters

        workflow_output_single = pd.DataFrame(columns=columns)
        for col in columns:
            for param in output_params:
                if col == param[0]:
                    workflow_output_single[col] = param[1]
            for param, value in input.items():
                if col == param:
                    workflow_output_single[col] = value

        workflow_output.append(workflow_output_single)

    return workflow_output


def active_train(dataframe):

    input_names = TWINLAB_PARAMS_TO_EMULATE.keys()
    output_names = WORKFLOW_OUTPUT_PARAMS

    dataset = tl.Dataset(id=GALAXY_WORKFLOW_NAME)

    # Add new data to model
    try:
        # previous_data = dataset.view()
        dataset.append(dataframe)
    except Exception as e:
        print(f'No previous data found: {e}')
        dataset.upload(dataframe)

    print(f'Inputs: {input_names}')
    print(f'Outputs: {output_names}')

    # Retrain model on added data
    try:
        print(dataset.view())
    except Exception as e:
        print(f"Couldn't print the dataset, error: {e}")
    emulator = tl.Emulator(id=GALAXY_WORKFLOW_NAME)
    params = tl.TrainParams(
        train_test_ratio=0.75,
        estimator="gaussian_process_regression",

    )
    emulator.train(
        dataset=dataset,
        inputs=input_names,
        outputs=output_names,
        params=params,
        verbose=True
    )

    return


def reccomend_training_points(num_points):
    emulator = tl.Emulator(GALAXY_WORKFLOW_NAME)

    params = tl.ReccommendParams(
        bounds=TWINLAB_PARAMS_TO_EMULATE
    )

    data = emulator.recommend(
        num_points=num_points,
        acq_func="explore",
        params=params
    )

    return data[0]


def model_evaluation():
    emulator = tl.Emulator(GALAXY_WORKFLOW_NAME)
    # validation step
    # calculate mean standardized log loss
    params = tl.ScoreParams(metric="MSLL", combined_score=True)
    MSLLscore = emulator.score(params=params)

    # calculate R squared
    params = tl.ScoreParams(metric="R2", combined_score=True)
    R2score = emulator.score(params=params)

    print(f"MSLL: {MSLLscore}, R2: {R2score}")

    # stopping criterion for active learning loop
    if MSLLscore < -1.5 and R2score > 0.95:
        print("model is good")
        return True
    else:
        print('Model not good enough, adding further training points')
        return False


if __name__ == "__main__":
    # Set up Galaxy
    gi = GalaxyInstance(url=GALAXY_SERVER_ADDR, key=GALAXY_API_KEY)

    # Set up TwinLab
    tl.set_api_key(TWINLAB_API_KEY)
    tl.set_user(TWINLAB_API_USER)
    initial_training_points = 5
    active_training_points = 5

    # Create initial training dataset
    initial_training = create_training_points(initial_training_points)

    # Run workflows on training dataset
    workflow_output = run_galaxy_workflows(gi, initial_training, initial_training_points)

    # Run training of model
    active_train(workflow_output)

    # Refinement loop
    while not model_evaluation():

        # Create new training points
        new_training = reccomend_training_points(active_training_points)

        # Run workflows on new training dataset
        workflow_output = run_galaxy_workflows(gi, new_training, active_training_points)

        # Run training of model
        active_train(workflow_output)
