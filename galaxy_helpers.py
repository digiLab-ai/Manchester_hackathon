from bioblend.galaxy import GalaxyInstance
import os

server = 'http://10.99.44.89:8080/'


def new_upload(gi, history, name, string):
    with open(name, 'w') as f:
        f.write(string)
    upload = gi.tools.upload_file(name, history)
    os.remove(name)
    return upload


def check_galaxy_api(api_key, uid):
    """
    Function to check if the provided API key is valid
    """

    # try:
    gi = GalaxyInstance(url=server, key=api_key)
    gi.workflows.get_workflows()
    return True


def check_workflow(api_key, workflow_name, uid):
    """
    Function to check if the workflow that is being referenced is part of
    the workflows available on the galaxy instance
    """

    workflow_galaxy = get_workflows(api_key, uid)

    if workflow_name not in workflow_galaxy:
        return False
    else:
        return True


def check_inputs(api_key, workflow_name, inputs, uid):
    """
    Function to check if the inputs provided are valid for the given workflow
    """
    expected_inputs = get_inputs(api_key, workflow_name, uid)

    if not expected_inputs:
        return False

    # Check if the inputs provided are the same as the expected inputs
    for expected_in in expected_inputs:
        if expected_in[1] not in inputs.keys():
            return False
    return True


def get_inputs(api_key, workflow_name, uid):
    """
    Function to get an array of inputs for a given galaxy workflow

    Usage:
        get_inputs(
            server = "http://mcfe.duckdns.org/",
            api_key = "ea8caa3beffee9c8d58c8b0a092d936e",
            workflow_name = "More Complex Test Workflow",
        )

    Args:
        server (string): Galaxy server address
        api_key (string): User generated string from galaxy instance
            to create: User > Preferences > Manage API Key > Create a new key
        workflow_name (string): Target workflow name
    Returns:
        inputs (array of strings): Input files expected by the workflow, these
        will be in the same order as they should be given in the main API call
    """

    # Check server and api key are valid
    if not check_galaxy_api(api_key, uid):
        return False
    # Check workflow exists
    if not check_workflow(api_key, workflow_name, uid):
        return False

    gi = GalaxyInstance(url=server, key=api_key)
    api_workflow = gi.workflows.get_workflows(name=workflow_name)
    steps = gi.workflows.export_workflow_dict(api_workflow[0]['id'])['steps']
    input_array = []
    for step in steps:
        inputs = steps[step]['inputs']
        name = steps[step]['name']

        # Some of the steps don't take inputs so have to skip these
        # And only pull the inputs from input datasets, not individual tools
        if len(inputs) > 0 and name == "Input dataset":
            for input in inputs:
                input_array.append(
                    ('dataset', input['name'], steps[step]['id'])
                )
        if len(inputs) > 0 and name == "Input parameter":
            for input in inputs:
                input_array.append(
                    ('parameter', input['name'], steps[step]['id'])
                )

    return input_array


def get_outputs(api_key, workflow_name, uid):
    """
    Function to get an array of outputs for a given galaxy workflow

    Usage:
        get_outputs(
            server = "http://mcfe.duckdns.org/",
            api_key = "ea8caa3beffee9c8d58c8b0a092d936e",
            workflow_name = "More Complex Test Workflow",
        )

    Args:
        server (string): Galaxy server address
        api_key (string): User generated string from galaxy instance
            to create: User > Preferences > Manage API Key > Create a new key
        workflow_name (string): Target workflow name
    Returns:
        outputs (array of strings): Output files given by the workflow,
            these are the names that can be requested as workflow outputs
    """

    # Check server and api key are valid
    if not check_galaxy_api(api_key, uid):
        return False
    if not check_workflow(api_key, workflow_name, uid):
        return False

    gi = GalaxyInstance(url=server, key=api_key)
    api_workflow = gi.workflows.get_workflows(name=workflow_name)
    steps = gi.workflows.export_workflow_dict(api_workflow[0]['id'])['steps']
    outputs = []

    for step in steps:
        # Some of the steps don't take inputs so have to skip these
        if not len(steps[step]) > 0:
            continue

        if 'outputs' not in steps[step]:
            continue

        output_dict = steps[step]['outputs']

        if not len(output_dict) > 0:
            continue

        # See if output has been renamed & grab that name instead
        if 'post_job_actions' in steps[step]:
            post_job_actions = steps[step]['post_job_actions']
            if 'RenameDatasetActionFile' in post_job_actions:
                action_file = post_job_actions['RenameDatasetActionFile']
                name = action_file['action_arguments']['newname']
                outputs.append(name)
                continue

        for output in output_dict:
            outputs.append(output['name'])

    return outputs


def get_workflows(api_key, uid):
    """
    Function to get an array of workflows available on a given galaxy instance

    Usage:
        get_workflows(
            config_dict
        )

    Args:
        get_workflows_config (dict): dictionary containing server and api_key
    Returns:
        workflows (array of strings): Workflows available to be run on the
            galaxy instance provided
    """
    # Check server and api key are valid
    if not check_galaxy_api(api_key, uid):
        return False

    gi = GalaxyInstance(url=server, key=api_key)
    workflows_dict = gi.workflows.get_workflows()
    workflows = []
    for item in workflows_dict:
        workflows.append(item['name'])
    return workflows


def get_parameter_outputs(api_key, workflow_name, uid):
    """
    Function to get an array of parameter outputs for a given galaxy workflow

    Usage:
        get_parameter_outputs(
            api_key = "ea8caa3beffee9c8d58c8b0a092d936e",
            workflow_name = "More Complex Test Workflow",
        )

    Args:
        api_key (string): User generated string from galaxy instance
            to create: User > Preferences > Manage API Key > Create a new key
        workflow_name (string): Target workflow name
    Returns:
        outputs (array of strings): Output files given by the workflow,
            these are the names that can be requested as workflow outputs
    """

    # Check server and api key are valid
    if not check_galaxy_api(api_key, uid):
        return False
    if not check_workflow(api_key, workflow_name, uid):
        return False

    gi = GalaxyInstance(url=server, key=api_key)
    api_workflow = gi.workflows.get_workflows(name=workflow_name)
    steps = gi.workflows.export_workflow_dict(api_workflow[0]['id'])['steps']

    outputs = {}

    for step in steps:
        # Some of the steps don't take inputs so have to skip these
        if not len(steps[step]) > 0:
            continue

        if len(steps[step]['workflow_outputs']) <= 0:
            continue

        for output in steps[step]['workflow_outputs']:
            name = output['output_name']
            if output['label'] is None:
                continue

            if '_float' in output['label']:
                real_name = output['label'].split('_float')[0]
                outputs[name] = ('float', real_name)
            elif '_integer' in output['label']:
                real_name = output['label'].split('_integer')[0]
                outputs[name] = ('integer', real_name)
            elif '_text' in output['label']:
                real_name = output['label'].split('_text')[0]
                outputs[name] = ('text', real_name)
            elif '_boolean' in output['label']:
                real_name = output['label'].split('_boolean')[0]
                outputs[name] = ('boolean', real_name)
            else:
                continue

    return outputs


def get_parameters_from_invocation(api_key, invocation_id, workflow_name, uid):
    '''
    Function to get the parameters from a galaxy invocation

    Args:
        api_key (string): User generated string from galaxy instance
            to create: User > Preferences > Manage API Key > Create a new key
        invocation_id (string): The id of the invocation to get the parameters
        workflow_name (string): Target workflow name
        uid (string): Unique identifier for the job

    Returns:
        parameters (array of tuples): The parameters from the invocation
            format: [(parameter_name, parameter_value), ...]
    '''
    gi = GalaxyInstance(url=server, key=api_key)

    # Get the parameter outputs for the invocation
    job = gi.jobs.get_jobs(invocation_id=invocation_id)
    outputs = gi.jobs.get_outputs(job[0]['id'])
    parameter_names = get_parameter_outputs(api_key, workflow_name, uid)

    # Get the dataset ids for the output parameters and download them
    # then convert to the correct type & write out
    return_params = {}
    for output in outputs:
        print(output)
        name = output['name']
        if name in parameter_names.keys():
            dataset_id = output['dataset']['id']
            type = parameter_names[name][0]
            name = parameter_names[name][1]
            download = gi.datasets.download_dataset(dataset_id=dataset_id)

            # For an empty or failed dataset
            if len(download) == 0:
                return_params.append((name, None))
                continue

            # For a normal dataset
            if type == 'float':
                parameter = float(download)
            elif type == 'integer':
                parameter = int(download)
            elif type == 'text':
                parameter = str(download)
            elif type == 'boolean':
                parameter = bool(download)
            else:
                raise ValueError(f'Dont know type {type}')
            return_params[name] = parameter

    return return_params

