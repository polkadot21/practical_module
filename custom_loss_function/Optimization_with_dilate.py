import torch
import numpy as np
from matplotlib import pyplot as plt
import json
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



with open('config.json') as f:
    config = json.load(f)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Optimization_with_dilate:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x).view([config['training']['batch_size'], 1, 1]).to(device)

        # Computes loss
        loss, loss_shape, loss_temporal = self.loss_fn(y, yhat, config['loss']['alpha'], config['loss']['gamma'], device)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    #define training

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=config['training']['n_epochs'], n_features=config['model']['input_dim']):
        model_path = 'saved/model.h5'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, 1, n_features]).to(device)
                #x_batch = x_batch.view([batch_size, 1, 1]).to(device)
                y_batch = y_batch.view([batch_size, 1, 1]).to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, 1, n_features]).to(device)
                    y_val = y_val.view([batch_size, 1, 1]).to(device)
                    #y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val).view([batch_size, 1, 1]).to(device)
                    val_loss = self.loss_fn(y_val, yhat, config['loss']['alpha'], config['loss']['gamma'], device)
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)


    #define evaluation

    def evaluate(self, test_loader, batch_size=1, n_features=config['model']['input_dim']):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.view([batch_size, 1, 1]).to(device)
                #y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values


    def predict(self, x, batch_size = 1, n_features = config['model']['input_dim']):
        with torch.no_grad():
            x = x.view([batch_size, 1, n_features]).to(device)
            self.model.eval()
            yhat = self.model(x)
        return yhat.to(device).detach().numpy()

    # plot loses
    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()