import logging
import os
import azureml.core

from azureml.core import Experiment
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

import src.aml_pipelines.pipelines_config as cfg

from src.aml_pipelines.env_variables import Env
from src.aml_pipelines.utils.azureml import get_compute_target, get_workspace
from src.aml_pipelines.utils.experiments import create_run_configuration
from src.aml_steps.utils.experiment_config import ExperimentConfigurationWrapper


# print version of loaded libraries
print("Loaded libraries:")
print("- azureml.core, version: ", azureml.core.VERSION)


def main():
    logging.warning("Loading environment variables...")
    e = Env()
    e.load_environment_variables(env_file_path='local.env', fallback_to_os=True)

    # Get Azure machine learning workspace
    logging.warning("Getting reference to existing Azure Machine Learning workspace...")
    auth = InteractiveLoginAuthentication(tenant_id=e.tenant_id)
    ws = get_workspace(e.workspace_name, auth, e.subscription_id, e.resource_group)

    # Get compute target. It has to be a GPU compute as such unit is requested by the 'Feature Extraction Inference' step
    compute_target = get_compute_target(ws, compute_name=e.gpu_compute_name, vm_size=e.gpu_vm_size)

    # Create run configuration
    run_config = create_run_configuration(ws)

    # -------
    # Step 1
    # -------

    # Define input 'prepared datasets'
    input_prepared_datasets = []
    experiment_configuration = ExperimentConfigurationWrapper()
    experiment_configuration.load(os.path.join(cfg.StepsStructure.SNAPSHOT_ROOT_DIR, cfg.StepsStructure.get_experiments_config_filepath()))  # noqa: E501
    for data_config in experiment_configuration.json['OBJECT_DETECTION']['inference']['data']:
        dataset_name = data_config['input']['dataset_name']
        dataset = ws.datasets.get(dataset_name)
        input_prepared_datasets.extend([dataset.as_named_input(dataset_name)])

    # Create pipeline datastore objects to create links between steps, so they are executed in a sequence, not in parallel
    pipeline_datastore = ws.get_default_datastore()
    object_detection_inference_output = PipelineData(name="centers", datastore=pipeline_datastore, is_directory=True)

    step_object_detection_inference = PythonScriptStep(
        name="Object Detection - Inference",
        source_directory=cfg.StepsStructure.SNAPSHOT_ROOT_DIR,
        script_name=cfg.StepsStructure.ObjectDetection.INFERENCE_STEP_SCRIPT_PATH,
        arguments=[
            '--subscription_id', e.subscription_id,
            '--resource_group', e.resource_group,
            '--workspace_name', e.workspace_name,
            '--experiments_config_filepath', cfg.StepsStructure.get_experiments_config_filepath(),
            '--model_name', cfg.MLModelNames.OBJECT_DETECTION_MODEL,
            '--model_version', cfg.MLModelNames.OBJECT_DETECTION_MODEL_BEST_VERSION,
            '--output_folder', object_detection_inference_output,
            '--should_register_dataset', True
        ],
        inputs=input_prepared_datasets,
        outputs=[object_detection_inference_output],
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=True
    )

    # -------
    # Step 2
    # -------

    # input should contain 'prepared datasets' and centers
    object_extraction_input = object_detection_inference_output.as_input('centers')
    object_extraction_inputs = [object_extraction_input]

    object_extraction_output = PipelineData(name="cropped_objects", datastore=pipeline_datastore, is_directory=True)

    step_object_extraction = PythonScriptStep(
        name="Object Extraction",
        source_directory=cfg.StepsStructure.SNAPSHOT_ROOT_DIR,
        script_name=cfg.StepsStructure.ObjectExtraction.STEP_SCRIPT_PATH,
        arguments=[
            "--subscription_id", e.subscription_id,
            "--resource_group", e.resource_group,
            "--workspace_name", e.workspace_name,
            "--experiments_config_filepath", cfg.StepsStructure.get_experiments_config_filepath(),
            "--output_folder", object_extraction_output,
            "--should_register_dataset", True,
            # This flag might be handy when we really want to recreate a cropped objects dataset (e.g. changed implementation
            # of the NucleiExtractor, although there are no changes in the input datasets).
            "--force_dataset_recreation", True
        ],
        inputs=object_extraction_inputs,
        outputs=[object_extraction_output],
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=True,
    )

    # -------
    # Step 3a
    # -------

    step_object_images_upload = PythonScriptStep(
        name="Cropped Object Images Upload to Blob Storage",
        source_directory=cfg.StepsStructure.SNAPSHOT_ROOT_DIR,
        script_name=cfg.StepsStructure.ObjectImagesUpload.STEP_SCRIPT_PATH,
        arguments=[
            # '--subscription_id', e.subscription_id,
            # '--resource_group', e.resource_group,
            # '--workspace_name', e.workspace_name,
            '--experiments_config_filepath', cfg.StepsStructure.get_experiments_config_filepath(),
            # '--model_name', cfg.MLModelNames.FEATURE_EXTRACTION_MODEL,
            # '--model_version', cfg.MLModelNames.FEATURE_EXTRACTION_MODEL_BEST_VERSION,
            # '--output_folder', feature_extraction_inference_output,
            # '--should_register_dataset', True
        ],
        inputs=[object_extraction_output.as_input('cropped_objects')],
        outputs=[],
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=True
    )

    # -------
    # Step 3b
    # -------

    feature_extraction_inference_input = object_extraction_output.as_input('cropped_objects')
    feature_extraction_inference_inputs = [feature_extraction_inference_input]

    feature_extraction_inference_output = PipelineData(name="latent_dims", datastore=pipeline_datastore, is_directory=True)

    step_feature_extraction_inference = PythonScriptStep(
        name="Feature Extraction - Inference",
        source_directory=cfg.StepsStructure.SNAPSHOT_ROOT_DIR,
        script_name=cfg.StepsStructure.FeatureExtraction.INFERENCE_STEP_SCRIPT_PATH,
        arguments=[
            '--subscription_id', e.subscription_id,
            '--resource_group', e.resource_group,
            '--workspace_name', e.workspace_name,
            '--experiments_config_filepath', cfg.StepsStructure.get_experiments_config_filepath(),
            '--model_name', cfg.MLModelNames.FEATURE_EXTRACTION_MODEL,
            '--model_version', cfg.MLModelNames.FEATURE_EXTRACTION_MODEL_BEST_VERSION,
            '--output_folder', feature_extraction_inference_output,
            '--should_register_dataset', True
        ],
        inputs=feature_extraction_inference_inputs,
        outputs=[feature_extraction_inference_output],
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=True
    )

    # -------
    # Pipeline composition
    # -------

    pipeline_steps = [
        step_object_detection_inference,
        step_object_extraction,
        step_object_images_upload,
        step_feature_extraction_inference
    ]
    pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

    # Create and submit an experiment
    logging.warning("Submitting experiment...")
    experiment = Experiment(ws, cfg.ExperimentNames.INFERENCE_REMOTE)
    experiment.submit(pipeline, regenerate_outputs=False)  # Allow data reuse for this run
    logging.warning('Experiment submitted!')


if __name__ == "__main__":
    main()
