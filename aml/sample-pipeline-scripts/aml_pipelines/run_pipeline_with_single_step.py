import azureml.core
import logging

from azureml.core import Experiment
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

import src.aml_pipelines.pipelines_config as cfg

from src.aml_pipelines.env_variables import Env
from src.aml_pipelines.utils.azureml import get_compute_target, get_workspace
from src.aml_pipelines.utils.experiments import create_run_configuration
from src.aml_pipelines.utils.print_nicely import print_green

# print version of loaded libraries
print("Loaded libraries:")
print("- azureml.core, version: ", azureml.core.VERSION)


def main():
    logging.warning("Loading environment variables...")
    e = Env()
    e.load_environment_variables(env_file_path="local.env")

    # Get Azure machine learning workspace
    logging.warning("Getting reference to existing Azure Machine Learning workspace...")
    auth = InteractiveLoginAuthentication(tenant_id=e.tenant_id)
    ws = get_workspace(e.workspace_name, auth, e.subscription_id, e.resource_group)

    # Get compute target.
    # The compute target is explicitely specified here to mitigate risk of choosing an incorrect machine, that would execute
    # heavy load experiments by triggering pipeline via REST API.
    compute_target = get_compute_target(ws, compute_name='cpu-high-load', vm_size='STANDARD_F64S_V2')

    # Create pipeline datastore
    pipeline_datastore = ws.get_default_datastore()
    step_output = PipelineData(
        name="step_output_data",
        datastore=pipeline_datastore,
        is_directory=True,
    )

    # Create run configuration
    run_config = create_run_configuration(ws)

    latent_dim_dataset_name_param = PipelineParameter(name="latent_dim_dataset_name", default_value='')
    channel_names_param = PipelineParameter(name="channels_names", default_value='')
    hyperparameters_param = PipelineParameter(name="hyperparameters", default_value='{}')

    # Define step
    step = PythonScriptStep(
        name=cfg.ExperimentNames.DIM_REDUCTION_REMOTE,
        source_directory=cfg.StepsStructure.SNAPSHOT_ROOT_DIR,
        script_name=cfg.StepsStructure.DimReduction.STEP_SCRIPT_PATH,
        arguments=[
            '--latent_dim_dataset_name', latent_dim_dataset_name_param,
            '--hyperparameters', hyperparameters_param,
            '--channels_names', channel_names_param,
            '--experiments_config_filepath', cfg.StepsStructure.get_experiments_config_filepath(),
            '--output_folder', step_output
        ],
        inputs=[],
        outputs=[step_output],
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=False
    )

    pipeline_steps = [step]
    pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

    # Create and submit an experiment
    logging.warning("Submitting experiment...(v0003)")
    experiment = Experiment(ws, cfg.ExperimentNames.DIM_REDUCTION_REMOTE)
    experiment.submit(
        pipeline,
        pipeline_parameters={
            "latent_dim_dataset_name": 'dataset_001',
            # TODO: default channel names should be taken from the config.json file.
            "channels_names": "a,b,c,d",
            # TODO: default hyperparameters should be taken from the config.json file.
            "hyperparameters": "{ 'a': 10, 'b': 0.0 }"
        },
        regenerate_outputs=False)  # Allow data reuse for this run
    print_green('Experiment submitted!')


if __name__ == "__main__":
    main()

