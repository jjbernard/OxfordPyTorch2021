import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import deepant_pytorch as dp
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import MeanAbsoluteError, MeanSquaredError, Loss
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import ProgressBar

func = dp.create_sinusoidal_time_series
total_length = 100000
dev = torch.device(dp.check_cuda())
series = func(total_length)

window = 45
dimensions = 1
prediction_window = 1
batch_size = 32
learning_rate = 0.001
epochs = 10
test_train_split = 0.8

train, test = dp.split_data(series, train_percent_size=test_train_split)

train_ds = dp.SimpleTSDataset(data=train, sequence_size=window, prediction_size=prediction_window)
test_ds = dp.SimpleTSDataset(data=test, sequence_size=window, prediction_size=prediction_window)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

model = dp.DeepAnt(window=window, p_window=prediction_window, dimensions=dimensions)
criterion = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train_step(engine, batch):
    x, y = batch
    x = x.to(dev)
    y = y.to(dev)

    model.train()
    x = x.requires_grad_()
    result = model(x)
    loss = criterion(result, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def test_step(engine, batch):
    x, y = batch

    model.eval()
    with torch.no_grad():
        x = x.to(dev)
        y = y.to(dev)

        result = model(x)

        return result, y


trainer = Engine(train_step)
tester = Engine(test_step)


@trainer.on(Events.STARTED)
def start_message():
    print("Training now!")


@trainer.on(Events.COMPLETED)
def done_message():
    print("Training done!")


# Or directly as a lambda function: see below
# trainer.add_event_handler(Events.COMPLETED, lambda _: print("Training done!"))

@trainer.on(Events.EPOCH_COMPLETED)
def run_test():
    tester.run(test_dl)


test_metrics = {"MAE": MeanAbsoluteError(), "MSE": MeanSquaredError(), "Loss": Loss(criterion)}

for name, metric in test_metrics.items():
    metric.attach(tester, name)

train_evaluator = create_supervised_evaluator(model, metrics=test_metrics, device=dev)


@trainer.on(Events.EPOCH_COMPLETED)
def run_validation():
    train_evaluator.run(train_dl)


@train_evaluator.on(Events.COMPLETED)
def log_train_results():
    metrics = train_evaluator.state.metrics
    mae = metrics["MAE"]
    mse = metrics["MSE"]
    loss = metrics["Loss"]
    print(f"Training results for Epoch: {trainer.state.epoch} - MAE: {mae:.3f} - MSE: {mse:.3f} - Loss: {loss:.3f}")


@tester.on(Events.COMPLETED)
def show_test_results():
    metrics = tester.state.metrics
    mae = metrics["MAE"]
    mse = metrics["MSE"]
    loss = metrics["Loss"]

    print(f"Test results for Epoch {trainer.state.epoch} - MAE: {mae:.3f} - MSE: {mse:.3f} - Loss: {loss:.3f}")


# Let's define the anomaly score function
def anomaly_score(engine):
    return engine.state.metrics["MAE"]


# Checkpoint to store n_saved best models wrt score function
checkpoint = ModelCheckpoint(
    "deepant-model-w-ignite",
    require_empty=False,
    n_saved=2,
    filename_prefix="best",
    score_function=anomaly_score,
    score_name="mae",
    global_step_transform=global_step_from_engine(trainer),
)

tester.add_event_handler(Events.COMPLETED, checkpoint, {"model": model})

ProgressBar().attach(trainer, output_transform=lambda x: {'batch loss': x})

trainer.run(train_dl, max_epochs=epochs)
