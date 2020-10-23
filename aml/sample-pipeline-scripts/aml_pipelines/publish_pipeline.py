import azureml.core

from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.core import PipelineRun, PipelineEndpoint

import src.aml_pipelines.pipelines_config as cfg

from src.aml_pipelines.env_variables import Env
from src.aml_pipelines.utils.azureml import get_workspace


# print version of loaded libraries
print("Loaded libraries:")
print("- azureml.core, version: ", azureml.core.VERSION)


def main():
    ############################
    # specify parameter values
    ############################
    run_id = None
    pipeline_version = '1.0'
    pipeline_endpoint_version = f'{pipeline_version} - DEV'

    # Make sure we can proceed with pipeline execution
    assert run_id is not None

    pipeline_name = cfg.PipelineEndpointNames.DIM_REDUCTION
    pipeline_endpoint_name = f"{pipeline_name} - version {pipeline_endpoint_version}"
    description = 'Dimension reduction (UMAP and HDBSCAN)'

    # Load environment variables
    print("Loading environment variables...")
    e = Env()
    e.load_environment_variables(env_file_path="local.env")

    # Get Azure machine learning workspace
    print("Getting reference to existing Azure Machine Learning workspace...")
    auth = InteractiveLoginAuthentication(tenant_id=e.tenant_id)
    ws = get_workspace(e.workspace_name, auth, e.subscription_id, e.resource_group)

    # Get specific run of the experiment, make sure it was completed successfully
    experiment_name = cfg.ExperimentNames.DIM_REDUCTION_REMOTE
    pipeline_experiment = ws.experiments.get(experiment_name)
    pipeline_run = PipelineRun(pipeline_experiment, run_id=run_id)
    if pipeline_run.status != 'Completed':
        msg = f"Pipeline run: {run_id} has the status: {pipeline_run.status}." + \
            " It's a good practise to publish only successfully completed pipelines runs."
        raise Exception(msg)

    # Publish pipeline...
    newly_published_pipeline = pipeline_run.publish_pipeline(name=pipeline_name, description=description,
                                                             version=pipeline_version)
    # ... and attach it to a new (or existing) pipeline endpoint
    try:
        pipeline_endpoint = PipelineEndpoint.get(workspace=ws, name=pipeline_endpoint_name)
        pipeline_endpoint_found = True
    except Exception:
        pipeline_endpoint_found = False
    finally:
        if pipeline_endpoint_found:
            pipeline_endpoint.add_default(newly_published_pipeline)
        else:
            pipeline_endpoint = PipelineEndpoint.publish(workspace=ws, name=pipeline_endpoint_name,
                                                         description=description, pipeline=newly_published_pipeline)
        # Link PublishedPipeline with PipelineRun by setting tag on the run that was just published.
        # As of today, as far as I know, there is direct way to see what run_id was used to publish pipeline.
        pipeline_run.tag(pipeline_endpoint_name, pipeline_endpoint.default_version)

    print("Done!")


if __name__ == "__main__":
    main()

