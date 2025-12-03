import os
from tqdm import tqdm
from pathlib import Path
import datetime
import argparse

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import matplotlib.pyplot as plt

from modelDataset import YTDataset
from model import ThumbnailModel
from printHelpers import printANSI
from checkpoint import loadLatestCheckpoint, saveLatestCheckpoint

IMAGE_SIZE = (640, 360)

# region arguments
parser = argparse.ArgumentParser(
    prog="python3 modelTraining.py",
    description="Trains a neural network on YouTube data",
    epilog="Copyright Liam Hillery, 2025",
)

parser.add_argument_group("File I/O")
parser.add_argument("-i", "--imageDir", default="./output/thumbnails", type=Path)
parser.add_argument("-o", "--outDir", default="./output", type=Path)

parser.add_argument_group("Training options")
parser.add_argument("-e", "--epochs", default=5, type=int)
parser.add_argument("--validation_epochs", default=None, type=int)
# endregion


# region trainOnce
# epoch function based on the default provided by pytorch
def trainOnce(
    model: torch.nn.Module,
    dataLoader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.Any,
    device: torch.device,
):
    running_loss = 0.0

    batch = 1

    for inputs, groundTruth in tqdm(iter(dataLoader)):
        inputs = inputs.to(device)
        groundTruth = groundTruth.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output = model(inputs)

        # print(output)
        # print(groundTruth)

        # Compute the loss and its gradients
        loss = criterion(output, groundTruth)
        loss.backward()
        # print(loss)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        batch += 1
        # print(f"batch {batch} / {len(dataLoader)} | training loss: {loss.item()}")

    scheduler.step()

    return running_loss / len(dataLoader)


# endregion


# region train
def train(
    config: dict,
    epochs: int,
    trainingSet: Dataset,
    valSet: Dataset,
    trialDir: Path,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
):
    model = ThumbnailModel(*IMAGE_SIZE).to(device)

    trainingDataLoader = DataLoader(
        trainingSet, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0
    )
    valDataLoader = DataLoader(
        valSet, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0
    )

    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()

    # optimizer = SGD(model.parameters(), lr=5e-6, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=float(config["lr"]))

    scheduler = ExponentialLR(optimizer, gamma=float(config["gamma"]))

    checkpointDir: Path = trialDir / "checkpoints"
    os.makedirs(checkpointDir, exist_ok=True)
    checkpoint = loadLatestCheckpoint(checkpointDir)

    if checkpoint:
        printANSI(f"loading checkpoint from {checkpointDir}...", "bold", "cyan")
        model.load_state_dict(checkpoint["model_parameters"])
        optimizer.load_state_dict(checkpoint["optimizer_parameters"])
        scheduler.load_state_dict(checkpoint["scheduler_parameters"])
        losses = checkpoint["losses"]
        val_losses = checkpoint["val_losses"]
        starting_epoch = checkpoint["epoch"] + 1
    else:
        losses = []
        val_losses = []
        starting_epoch = 1

    # TODO THE REASON TESTING < TRAINING IS THAT ITS ONLY TAKEN AT THE BEST ITERATION
    # training loop, based on the one provided by pytorch
    for epoch in range(starting_epoch, epochs + 1):
        printANSI(f"-=-=-=-=- EPOCH {epoch} -=-=-=-=-", "bold", "yellow")

        print("training...")
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        avg_loss = trainOnce(
            model, trainingDataLoader, criterion, optimizer, scheduler, device
        )
        losses.append(avg_loss)

        running_val_loss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        print("validating...")
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for val_inputs, val_labels in tqdm(iter(valDataLoader)):
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                running_val_loss += val_loss

        avg_val_loss = running_val_loss / (len(valDataLoader) + 1)
        val_losses.append(avg_val_loss)

        # report losses
        print(f"Training loss (MSE): {avg_loss}")
        print(f"Testing loss (MSE): {avg_val_loss}")

        # finally, save the model params for future reference
        checkpoint = {
            "model_parameters": model.state_dict(),
            "optimizer_parameters": optimizer.state_dict(),
            "scheduler_parameters": scheduler.state_dict(),
            "losses": losses,
            "val_losses": [loss.cpu() for loss in val_losses],
            "epoch": epoch,
        }
        saveLatestCheckpoint(checkpointDir, checkpoint, epoch)

    # retrieve losses from gpu
    val_losses = [loss.cpu() for loss in val_losses]

    with open(
        trialDir / "losses.csv",
        "w",
    ) as f:
        fileBody = "loss,val loss\n"
        for i in range(len(losses)):
            fileBody += f"{losses[i]},{val_losses[i]}\n"
        f.write(fileBody)

    return losses, val_losses


# endregion


# region Main Execution
def main(argv=None):
    args = parser.parse_args(argv)
    currentTime = datetime.datetime.now()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator().manual_seed(1)

    printANSI(f"running torch on {device}", "bold", "bright_magenta")

    dataset = YTDataset(args.imageDir, IMAGE_SIZE)
    trainingSet, testingSet = random_split(dataset, (0.85, 0.15), generator=rng)

    # testingDataLoader = DataLoader(
    #     testingSet, batch_size=args.batch_size, shuffle=True, num_workers=0
    # )

    # TODO: add flag for displaying these things
    # gridSize = (4, 6)
    # index = np.random.randint(0, len(dataset) - np.prod(gridSize))
    # showGrid(gridSize, dataset[index:index+np.prod(gridSize)][0])

    hyperparams = {"batch_size": 16, "lr": 1e-4, "gamma": 0.9}

    losses, v_losses = train(
        hyperparams,
        args.epochs,
        trainingSet,
        testingSet,
        args.outDir / f"session_{currentTime.isoformat()}",
    )

    fig = plt.figure()

    plt.plot(range(len(v_losses)), v_losses, label="validation", color="#fa96c8")
    plt.plot(range(len(losses)), losses, label="training", color="#6496fa")
    plt.legend()

    fig.show()
    fig.savefig(f"{args.outDir}/losses.png")


# endregion


# region Entry Point
if __name__ == "__main__":
    main()
# endregion
