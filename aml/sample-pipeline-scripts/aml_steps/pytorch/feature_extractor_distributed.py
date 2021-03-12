import argparse
# import h5py as h5
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch


from azureml.core import Dataset
from azureml.core.run import Run
from autoencoders import CnnAEsym
from data_loader import CellDataset, HDF5TrainTestSplitter
# from data_loader import NucleiExtractor
from model_trainer import ModelTrainer, ModelTrainerDistributed
from utils.azureml import register_model, download_registered_file_dataset
from utils.argparse import str2bool
from utils.experiment_config import ExperimentConfigurationWrapper
from utils.file import list_all_files_in_location

import horovod.torch as hvd

hvd.init()


logging.basicConfig(level=logging.DEBUG)


DATA_LOADER_WORKERS = 8
DEVICE = "cuda:0"
LATENT_DIM_SIZE = 100
INPUT_IMAGE_SIZE = (64, 64)
EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
RANDOM_SEED = 1
BBOX_SIZE_ENLRAGEMENT = 10
AUTOENCODER = CnnAEsym


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def main():
    logging.warning("dummy warning!!!")
    logging.error("dummy error!!!")
    logging.info("dummy info!!!")
    logging.debug("dummy debug!!!")

    logging.warning(f"Inside {__file__}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription_id", type=str, dest="subscription_id", help="The Azure subscription ID")
    parser.add_argument("--resource_group", type=str, dest="resource_group", help="The resource group name")
    parser.add_argument("--workspace_name", type=str, dest="workspace_name", help="The workspace name")
    parser.add_argument("--experiments_config_filepath", type=str, dest="experiments_config_filepath", help="A path to the JSON config file")  # noqa: E501
    parser.add_argument("--model_name", type=str, dest="model_name", help="Name of the Model")
    parser.add_argument("--should_register_model", type=str2bool, dest="should_register_model", default=False, help="Register trained model")  # noqa: E501
    parser.add_argument("--use-cuda", type=str2bool, dest="use_cuda", default=True, help="Use CUDA for training?")
    args = parser.parse_args()

    logging.warning(f"Argument 1: {args.subscription_id}")
    logging.warning(f"Argument 2: {args.resource_group}")
    logging.warning(f"Argument 3: {args.workspace_name}")
    logging.warning(f"Argument 4: {args.experiments_config_filepath}")
    logging.warning(f"Argument 5: {args.model_name}")
    logging.warning(f"Argument 6: {args.should_register_model}")

    run = Run.get_context()
    
    if args.use_cuda:
        torch.cuda.set_device(hvd.local_rank())

    # Get current service context
    workspace = run.experiment.workspace

    # Load training configuration
    experiment_configuration = ExperimentConfigurationWrapper()
    experiment_configuration.load(args.experiments_config_filepath)
    training_config = experiment_configuration.json["feature_extractor"]["training"]

    download_root_dir = os.path.join('/mnt', 'tmp', 'datasets')
    data_splitter = HDF5TrainTestSplitter()
    for data_config in training_config["data"]:
        cropped_cells_dataset_name = data_config['input']['cropped_cells_dataset_name']
        cropped_cells_dataset_version = data_config['input']['cropped_cells_dataset_version']
        cropped_cells_dataset = Dataset.get_by_name(workspace=workspace, name=cropped_cells_dataset_name,
                                                    version=cropped_cells_dataset_version)

        msg = (f"Dataset '{cropped_cells_dataset_name}', id: {cropped_cells_dataset.id}"
               f", version: {cropped_cells_dataset.version} will be used to prepare data for a feature extractor training.")
        logging.warning(msg)

        # Create a folder where datasets will be downloaded to
        dataset_target_path = os.path.join(download_root_dir, cropped_cells_dataset_name)
        os.makedirs(dataset_target_path, exist_ok=True)

        # Download 'cropped cells' dataset (consisting of HDF5 and CSV files)
        dataset_target_path = download_registered_file_dataset(workspace, cropped_cells_dataset, download_root_dir)
        list_all_files_in_location(dataset_target_path)

        # Split data (indices) into subsets
        df_metadata = pd.read_csv(os.path.join(dataset_target_path, 'cropped_nuclei.csv'))
        logging.warning(f"Metadata dataframe (shape): {df_metadata.shape}")

        logging.warning("Splitting data into subsets...")
        data_splitter.add_dataset(name=data_config['input']['cropped_cells_dataset_name'], 
                                    fname=os.path.join(dataset_target_path, 'cropped_nuclei_images.h5'),
                                    metadata=df_metadata)
    
    data_splitter.train_dev_test_split()

    # --------
    # Training
    # --------

    # Init dataloaders
    train_dataset = CellDataset(splitter=data_splitter, dset_type='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=DATA_LOADER_WORKERS,
    )

    dev_dataset = CellDataset(splitter=data_splitter, dset_type='dev')
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    dev_data_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        sampler=dev_sampler,
        shuffle=False,
        num_workers=DATA_LOADER_WORKERS,
    )
    
    test_dataset = CellDataset(splitter=data_splitter, dset_type='test')
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        shuffle=False,
        num_workers=DATA_LOADER_WORKERS,
    )

    # Define and Train model
    #device = torch.device(DEVICE)
    model = AUTOENCODER(
        latent_dim_size=LATENT_DIM_SIZE,
        input_image_size=INPUT_IMAGE_SIZE,
    )
    if args.use_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    trainer = ModelTrainerDistributed(model, optimizer, train_data_loader, train_sampler, dev_data_loader, dev_sampler, 
                                                                                            test_data_loader, test_sampler, hvd)
    tr_losses, dev_losses = trainer.train(
        epochs=EPOCHS
    )
    test_loss = metric_average(trainer.get_test_loss(), "test_loss")
    run.log("dev_loss", dev_losses[-1])
    run.log("train_loss", tr_losses[-1])
    run.log("test_loss", test_loss)
    
    # Training completed!  Let's save the model and upload it to AML
    os.makedirs("./models", exist_ok=True)
    model_file_name = "model.ext"
    model_output_loc = os.path.join(".", "models", model_file_name)

    # Run this only on the main horovod node
    if hvd.rank() == 0:
        # Plot training metrics and model sample reconstructions
        trainer.get_training_plot(tr_losses=tr_losses, dev_losses=dev_losses)
        run.log_image("model training metrics", plot=plt)

        # Plot sample reconstructions
        dataiter = iter(test_data_loader)
        images = dataiter.next()
        trainer.get_pred_samples(images, figsize=(40, 40))
        run.log_image("sample reconstructions", plot=plt)

        torch.save(model, model_output_loc)

        run.upload_files(names=[model_output_loc], paths=[model_output_loc])

        # Register model (ideally, this should be a separate step)
    if args.should_register_model:
        logging.warning("List of the associated stored files:")
        logging.warning(run.get_file_names())

        logging.warning("Registering a new model...")
        # TODO: prepare a list of metrics that were logged using run.log()
        metric_names = []

        if os.path.exists(model_output_loc):
            register_model(
                run=run,
                model_name=args.model_name,
                model_description="Feature extraction model",
                model_path=model_output_loc,
                training_context="PythonScript",
                metric_names=metric_names,
            )
        else:
            logging.warning(f"Cannot register model as path {model_output_loc} does not exist.")
    else:
        logging.warning("A trained model will not be registered.")

    logging.warning("Done!")
    logging.info("Done Info Style!")


if __name__ == "__main__":
    main()
