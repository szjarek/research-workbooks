import torch
from matplotlib import pyplot as plt
import numpy as np
import torchvision

import torch.nn as nn

import copy


class ModelTrainer:
    def __init__(self, model, device):
        self.model = nn.DataParallel(model)
        self.device = device
        self.model.to(device)

    def _run_epoch(self, data_loader, optimizer):
        self.model.train()
        tr_loss = 0
        for data in data_loader:
            inputs = data.to(self.device)
            loss = self.model.module.calc_loss(inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(data_loader.dataset)
        return tr_loss

    def test_model(self, data_loader):
        self.model.eval()
        ts_loss = 0
        with torch.no_grad():
            for data in data_loader:
                inputs = data.to(self.device)
                ts_loss += self.model.module.calc_loss(inputs).item()
        ts_loss /= len(data_loader.dataset)
        return ts_loss

    def train(self, epochs, optimizer, train_data_loader, dev_data_loader):
        tr_losses = []
        dev_losses = []
        best_model = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf
        for epoch in range(1, epochs + 1):
            epoch_train_loss = self._run_epoch(train_data_loader, optimizer)
            epoch_test_loss = self.test_model(dev_data_loader)
            print(
                f"epoch: {epoch}, training set loss: {epoch_train_loss}, development set loss {epoch_test_loss}"
            )
            tr_losses.append(epoch_train_loss)
            dev_losses.append(epoch_test_loss)
            if epoch_test_loss < best_loss:
                best_model = copy.deepcopy(self.model.state_dict())
                best_loss = epoch_test_loss
        self.model = copy.deepcopy(best_model.state_dict())
        return tr_losses, dev_losses

    def get_training_plot(self, tr_losses, dev_losses, figsize=(20, 20)):
        fig = plt.figure(figsize=figsize)
        plt.plot(tr_losses, label="train set loss")
        plt.plot(dev_losses, label="development set loss")
        plt.legend()
        return fig

    def get_pred_samples(self, input_data, row_samples_num=20, figsize=(20, 20)):
        x_pred, _ = self.model(input_data.to(self.device))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.axis("off")

        chunks_num = input_data.shape[0]
        chunks_orig = input_data.chunk(chunks_num)
        chunks_preds = x_pred.cpu().chunk(chunks_num)
        pair_wise_tensor = torch.Tensor()

        for img, pred in zip(chunks_orig, chunks_preds):
            pair_wise_tensor = torch.cat((pair_wise_tensor, img, pred))

        pred_grid_data = torchvision.utils.make_grid(
            pair_wise_tensor.cpu().data, nrow=row_samples_num
        )
        pred_npimg = pred_grid_data.numpy()
        ax.imshow(np.transpose(pred_npimg, (1, 2, 0)))
        return fig

    def get_latent_predictions(self, input_data):
        _, x_latent = self.model(input_data.to(self.device))
        return x_latent


class ModelTrainerDistributed:
    def __init__(self, model, optimizer, train_data_loader, train_sampler, 
                    dev_data_loader, dev_sampler, test_data_loader, test_sampler, hvd):
        self.model = model
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.train_sampler = train_sampler
        self.dev_data_loader = dev_data_loader
        self.dev_sampler = dev_sampler
        self.test_data_loader = test_data_loader
        self.test_sampler = test_sampler
        self.hvd = hvd

    def metric_average(self, val, name):
        tensor = torch.tensor(val)
        avg_tensor = self.hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def _run_epoch(self, epoch):
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        running_loss = 0.
        for batch_idx, data in enumerate(self.train_data_loader):
            inputs = data.cuda()
            loss = self.model.calc_loss(inputs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(self.train_sampler),
                100. * batch_idx / len(self.train_data_loader), loss.item()))
        tr_loss = running_loss / len(self.train_sampler)
        return tr_loss

    def get_dev_loss(self):
        self.model.eval()
        running_loss = 0.
        with torch.no_grad():
            for data in self.dev_data_loader:
                inputs = data.cuda()
                running_loss += self.model.calc_loss(inputs).item()
        dev_loss = running_loss / len(self.dev_sampler)
        return dev_loss

    def get_test_loss(self):
        self.model.eval()
        running_loss = 0.
        with torch.no_grad():
            for data in self.test_data_loader:
                inputs = data.cuda()
                running_loss += self.model.calc_loss(inputs).item()
        test_loss = running_loss / len(self.test_sampler)
        return test_loss

    def train(self, epochs):
        tr_losses = []
        dev_losses = []
        best_model = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf
        for epoch in range(1, epochs + 1):
            epoch_train_loss = self._run_epoch(epoch)
            epoch_dev_loss = self.get_dev_loss()
            print(
                f"epoch: {epoch}, training set loss: {epoch_train_loss}, development set loss {epoch_dev_loss}"
            )
            tr_losses.append(epoch_train_loss)
            dev_losses.append(epoch_dev_loss)
            if epoch_dev_loss < best_loss:
                print(f"Got a better model, updating...")
                best_model = copy.deepcopy(self.model.state_dict())
                best_loss = epoch_dev_loss
            else:
                print(f"No better model this time...")
        self.model.load_state_dict(best_model)
        return tr_losses, dev_losses

    def get_training_plot(self, tr_losses, dev_losses, figsize=(20, 20)):
        fig = plt.figure(figsize=figsize)
        plt.plot(tr_losses, label="train set loss")
        plt.plot(dev_losses, label="development set loss")
        plt.legend()
        return fig

    def get_pred_samples(self, input_data, row_samples_num=20, figsize=(20, 20)):
        x_pred, _ = self.model(input_data.cuda())
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.axis("off")

        chunks_num = input_data.shape[0]
        chunks_orig = input_data.chunk(chunks_num)
        chunks_preds = x_pred.cpu().chunk(chunks_num)
        pair_wise_tensor = torch.Tensor()

        for img, pred in zip(chunks_orig, chunks_preds):
            pair_wise_tensor = torch.cat((pair_wise_tensor, img, pred))

        pred_grid_data = torchvision.utils.make_grid(
            pair_wise_tensor.cpu().data, nrow=row_samples_num
        )
        pred_npimg = pred_grid_data.numpy()
        ax.imshow(np.transpose(pred_npimg, (1, 2, 0)))
        return fig

    def get_latent_predictions(self, input_data):
        _, x_latent = self.model(input_data.cuda())
        return x_latent
