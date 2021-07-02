import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler


def create_constant_time_series(nb_steps: int) -> np.ndarray:
    return np.array([2] * nb_steps, dtype='float32').reshape(-1, 1)


def create_linear_time_series(nb_steps: int) -> np.ndarray:
    return np.linspace(1, nb_steps, num=nb_steps, dtype='float32').reshape(-1, 1)


def create_sinusoidal_time_series(nb_steps: int) -> np.ndarray:
    steps = np.linspace(0, 4, num=nb_steps)
    return np.sin(np.pi * steps, dtype='float32').reshape(-1, 1)


def check_cuda() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def normalize_data(data: np.ndarray) -> Tuple[MinMaxScaler, np.ndarray]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_norm = scaler.fit_transform(data)
    return scaler, data_norm


def split_data(data: np.ndarray, train_percent_size: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:

    length = data.shape[0]
    train_range = int(length * train_percent_size)

    train_data = data[:train_range, :]
    test_data = data[train_range:, :]

    return train_data, test_data


def calculate_size(L_in: int, padding: int, dilation: int, kernel_size: int, stride: int) -> int:
    res = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
    res = 1 + res / stride
    L_out = math.floor(res)
    return int(L_out)


class SimpleTSDataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_size: int = 5, prediction_size: int = 1,
                 device: torch.device = "cpu"):
        self.data = data

        # We assume the data is in the following form: (n, length) where n is the number of dimension
        # in the time series data (i.e. 1 for univariate and more than 1 for multivariate) and length
        # is the length of the time series.

        self.x = []
        self.y = []

        n_items = self.data.shape[0]
        n_dims = self.data.shape[1]
        self.max_range = n_items - sequence_size - prediction_size

        for i in range(self.max_range):
            x_temp = torch.from_numpy(self.data[i:i + sequence_size, :]).permute(1, 0).to(device)
            y_temp = torch.from_numpy(self.data[i + sequence_size:i + prediction_size + sequence_size, :]).permute(1,
                                                                                                                   0).to(
                device)
            # self.x.append(x_temp.reshape(-1, n_dims))
            self.x.append(x_temp)
            self.y.append(y_temp)

    def __len__(self) -> int:
        return self.max_range

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        item = (self.x[index], self.y[index])
        return item


class DeepAnt(nn.Module):
    def __init__(self, window: int = 5, p_window: int = 1, nb_filters: int = 32, kernel_conv_size: int = 3,
                 kernel_pool_size: int = 2, dimensions: int = 1):
        super(DeepAnt, self).__init__()
        self.window = window
        self.p_window = p_window

        end_size_conv1 = calculate_size(window, 1, 1, kernel_conv_size, 1)
        end_size_pool1 = calculate_size(end_size_conv1, 1, 1, kernel_pool_size, kernel_pool_size)
        end_size_conv2 = calculate_size(end_size_pool1, 1, 1, kernel_conv_size, 1)
        end_size_pool2 = calculate_size(end_size_conv2, 1, 1, kernel_pool_size, kernel_pool_size)

        self.conv1 = nn.Conv1d(dimensions, nb_filters, kernel_conv_size, padding=1, bias=True)
        self.conv2 = nn.Conv1d(nb_filters, nb_filters, kernel_conv_size, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_pool_size, padding=1)
        self.fc = nn.Linear(end_size_pool2 * nb_filters, p_window * dimensions)

    def forward(self, input):
        batch_size = input.size(0)
        intermediate = self.conv1(input)
        intermediate = self.relu(intermediate)
        intermediate = self.maxpool(intermediate)

        out = self.conv2(intermediate)
        out = self.relu(out)
        out = self.maxpool(out)

        # Todo: we should resize the output to (N, dimensions, p_window)
        result = self.fc(out.view(batch_size, -1))
        # result = self.relu(result)

        return torch.unsqueeze(result, 2)


if __name__ == '__main__':
    # func defines the function we use to create the time series
    # available choices:
    # * create_constant_time_series
    # * create_linear_time_series
    # * create_sinusoidal_time_series
    # Do NOT put brackets to the name of the function!

    func = create_sinusoidal_time_series
    total_length = 3000
    dev = torch.device(check_cuda())
    series = func(total_length)

    window = 45
    dimensions = 1
    prediction_window = 1
    batch_size = 32
    learning_rate = 0.001
    epochs = 100
    test_train_split = 0.9

    train, test = split_data(series, train_percent_size=test_train_split)

    train_ds = SimpleTSDataset(data=train, sequence_size=window, prediction_size=prediction_window)
    test_ds = SimpleTSDataset(data=test, sequence_size=window, prediction_size=prediction_window)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    model = DeepAnt(window=window, p_window=prediction_window, dimensions=dimensions)
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = list()

    for epoch in range(epochs):
        count = 0
        total_loss = 0
        for x, y in train_dl:
            model.train()
            x = x.requires_grad_()

            optimizer.zero_grad()
            result = model(x)
            loss = criterion(result, y)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            count += 1

        losses.append(total_loss / count)

    plt.plot(np.array(losses))
    plt.title("Training Loss")
    plt.show()

    print("Evaluation")
    results = []
    model.eval()
    for x, y in test_dl:
        result = model(x)
        results.append(result)

    reconstruct = []
    for data in results:
        data = data[:, 0, :]
        res = data.reshape(-1)
        res = res.detach().numpy()
        for r in res:
            reconstruct.append(r)

    plt.plot(np.array(reconstruct))
    plt.title("Output on test data")
    plt.show()
