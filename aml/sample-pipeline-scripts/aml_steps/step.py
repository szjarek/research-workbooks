import argparse
import logging
import os
from glob import glob
import json

from azureml.core.run import Run
from azureml.core import Dataset, Datastore

import dim_reduction

from utils.azureml import add_run_tag


# TODO: Consider moving it to env (if nothing more important is left to do ;-))
DIMENSION_REDUCTION_RESULTS_STORE = "dimension_reduction_results_store"


def main():

    logging.warning(f"Inside {__file__}")

    parser = argparse.ArgumentParser()

    parser.add_argument("--latent_dim_dataset_name", type=str, dest='latent_dim_dataset_name')
    parser.add_argument("--channels_names", type=str, dest='channel_names', help='Comma-separated list of channels to concatenate')  # noqa: E501
    parser.add_argument("--hyperparameters", type=str, dest='hyperparameters', help='The dictionary of hyperparameters and their values passed as a Json object')  # noqa: E501
    parser.add_argument("--experiments_config_filepath", type=str, dest='experiment_config_filepath', help='A path to the JSON config file')  # noqa: E501
    parser.add_argument("--output_folder", type=str, dest="output_folder", default='./outputs', help="Folder where output results should be written to")  # noqa: E501
    args = parser.parse_args()

    print(f"Argument latent_dim_dataset_name: {args.latent_dim_dataset_name}")
    print(f"Argument channel_names: {args.channel_names}")
    print(f"Argument hyperparameters: {args.hyperparameters}")
    print(f"Argument experiment_config_filepath: {args.experiment_config_filepath}")
    print(f"Argument output_folder: {args.output_folder}")

    # Get current service context
    run = Run.get_context()
    ws = run.experiment.workspace

    print(f"Run id: {run.id}")
    print("Run type: {}".format(str(type(run))))

    # let all used hyperparameters be easily visible in the UI
    try:
        add_run_tag(run, 'DATASET_NAME', args.latent_dim_dataset_name)
        add_run_tag(run, 'CHANNEL_NAMES', args.channel_names)
        hyperparameters = json.loads(args.hyperparameters.upper().replace('\'', '"'))
        for key in hyperparameters.keys():
            add_run_tag(run, key, hyperparameters[key])
    except Exception as ex:
        logging.warning(f"Problem with tagging run. Details: {ex}")

    try:
        print("Run tags: {}".format(str(run.tags)))
        if run.parent is not None:
            print("Run tags: {}".format(str(run.parent.tags)))
    except Exception:
        logging.warning("No Run tags")

    # Load inference configuration
    with open(args.experiment_config_filepath) as f:
        experiments_configuration = json.load(f)
        config = experiments_configuration["dimension_reduction"]["inference"]

    try:
        config_dataset_name = config['data'][0]['input']['latent_dims_dataset_name']
        print("Config dataset name: [{}]".format(config_dataset_name))
    except Exception:
        logging.error("Missing dataset configuration in file %s", args.experiment_config_filepath)
        config_dataset_name = None

    print(f"Downloading datasets...")
    root_ds_dir = os.path.join('/mnt', 'tmp', 'datasets')
    print(f"Datasets root: {root_ds_dir}")

    # Downloading configured dataset

    dataset_name = args.latent_dim_dataset_name
    try:
        ds = Dataset.get_by_name(ws, name=dataset_name)
        print("Downloading dataset from input parameter:", dataset_name)
    except Exception:
        logging.error(
            "Could not find dataset: [%s] requested by the input parameter. Falling back to the default dataset stored in the configuration file.",  # noqa: E501
            dataset_name)
        dataset_name = config_dataset_name
        ds = Dataset.get_by_name(ws, name=dataset_name)
        logging.warning("Downloading dataset from configuration: %s", dataset_name)

    dataset_target_path = os.path.join(root_ds_dir, dataset_name)
    print(f"Target dataset path:", dataset_target_path)
    ds.download(target_path=dataset_target_path, overwrite=True)
    print("Dataset {} downloaded...".format(dataset_name))

    files = glob(os.path.join(dataset_target_path, '*.*'))
    print("Files downloaded:", len(files))

    print("Files:", str(files))

    print("Output folder:", args.output_folder)

    # Main processing logic runs here:
    print("Starting dimension reduction...")
    dim_reduction.run_inference(
        dataset_target_path,
        args.output_folder,
        hyperparameters_enc=args.hyperparameters,
        channels_enc=args.channel_names)
    print("Dimension reduction completed...")

    # Uploading data to blob datastore:
    print(f"Uploading files to result store...")
    try:
        res_datastore = Datastore.get(ws, DIMENSION_REDUCTION_RESULTS_STORE)
    except Exception as ex:
        logging.error("Could not obtain datastore: %s. Please make sure it is created and available for the current user. \nException: %s",  # noqa: E501
                      DIMENSION_REDUCTION_RESULTS_STORE, str(ex))

    # Setting target path to run id (Pipeline run id if available)
    target_path = run.tags.get("azureml.pipelinerunid", str(run.id))
    print("output target run id:", target_path)
    res_datastore.upload(args.output_folder, target_path=target_path, overwrite=False, show_progress=True)
    print(f"...COMPLETED")

    print("Done!")


if __name__ == "__main__":
    main()


