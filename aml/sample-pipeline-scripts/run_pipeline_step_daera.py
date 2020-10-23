import os
import argparse
from glob import glob
# from azureml.core import Dataset, Workspace
from azureml.core.run import Run

# import proj_compute.infrastructure.files as files_local
# import proj_compute.infrastructure.files_az as files_azure
# import proj_compute.logic.config as cfg

from proj_compute.infrastructure.config import read_config
from proj_compute.logic.pipeline_steps import run_pipeline_step

# TODO: Change logger behavior - despite setting on DEBUG mode, when run in context of AML, only warnings and
#       more severe messages are currently logged :-(
from proj_compute.logic.logging import LOGGER


def _run_step(step_name: str, input_root_path: str, output_root_path: str, experiment_config_path: str):
    read_config(input_root_path, output_root_path, experiment_config_path)

    files_local = None  # TODO: fix that
    rep_factory = files_local.LocalRepositoryFactory
    run_pipeline_step(step_name, rep_factory)


def main():
    """ Main pipeline step function. """

    # parse arguments
    parser = argparse.ArgumentParser(description="Runs selected pipeline step.")
    parser.add_argument('--input_root_path', '-i', type=str, required=False)
    parser.add_argument('--output_root_path', '-o', type=str, required=True)
    parser.add_argument('--input_dataset_name', '-ids', type=str, required=False)
    parser.add_argument('--experiment_config_path', '-e', type=str, required=True)
    parser.add_argument('--step_name', '-s', type=str, required=True)
    args = parser.parse_args()

    # Make sure output folder exists
    LOGGER.info("Ensuring folder for output: %s", args.output_root_path)
    LOGGER.debug("Output as directory already exists: %s", os.path.isdir(args.output_root_path))
    LOGGER.debug("Output as file already exists: %s", os.path.isfile(args.output_root_path))
    os.makedirs(args.output_root_path, exist_ok=True)

    if (args.input_dataset_name is not None):
        LOGGER.info("Dataset_name provided, ignoring input_root_path, mapping registered dataset as input...")
        # Get current service context
        run = Run.get_context()
        if run.input_datasets:
            dataset = run.input_datasets[args.input_dataset_name]
        else:
            raise Exception("Cound not find dataset in passed inputs", args.input_dataset_name)

        with dataset.mount(mount_point=os.path.join('/srv/data', args.input_dataset_name)) as mount_context:
            # list top level mounted files and folders in the dataset
            LOGGER.debug("AZURE_DEBUG: List dir: %s", mount_context.mount_point)
            # LOGGER.warning("AZURE_DEBUG: Files: %s", str(os.listdir(mount_context.mount_point)))

            _run_step(args.step_name, mount_context.mount_point, args.output_root_path, args.experiment_config_path)

            output = [os.path.split(f)[1] for f in glob(os.path.join(args.output_root_path, "*"))]
            if len(output) == 0:
                LOGGER.warning("No files in output! Please investigate the step logs for more details.")
            # LOGGER.warning("AZURE_DEBUG: Output files: %s", str(output))

    else:
        output_root_path = args.output_root_path
        if output_root_path is None:
            output_root_path = args.input_root_path

        LOGGER.info("No dataset_name provided, using input/output paths...")
        _run_step(args.step_name, args.input_root_path, args.output_root_path, args.experiment_config_path)


if __name__ == '__main__':
    main()
