import os
from tqdm import tqdm
import argparse
import pickle
from functools import partial

# import matplotlib.pyplot as plt
import pathlib

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import ray.tune
import ray.train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler

from modelDataset import YTDataset
from model import ThumbnailModel, getNumParams, resetAllWeights
from printHelpers import printANSI, printBox

# region arguments
parser = argparse.ArgumentParser(
    prog="python3 modelTraining.py",
    description="Trains a neural network on YouTube data",
    epilog="Copyright Liam Hillery, 2025",
)

parser.add_argument_group("File I/O")
parser.add_argument(
    "-i", "--imageDir", default="./output/thumbnails", type=pathlib.Path
)
parser.add_argument("-o", "--outDir", default="./output", type=pathlib.Path)

# TODO: change to checkpoint
parser.add_argument("-s", "--starting-weights", default=None, type=pathlib.Path)

parser.add_argument_group("Training options")
parser.add_argument("-e", "--epochs", default=5, type=int)
parser.add_argument("-b", "--batch-size", default=20, type=int)
parser.add_argument(
    "-v", "--validation-only", action="store_true"
)  # TODO: not yet implemented
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
    hyperparams: dict,
    model: torch.nn.Module,
    trainingDataset: Dataset,
    epochs: int,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
    outDir: pathlib.Path = None,
):
    if ray.tune.get_context().get_experiment_name():
        outDir /= ray.tune.get_context().get_experiment_name()

    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()

    # optimizer = SGD(model.parameters(), lr=5e-6, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=hyperparams["lr"])

    scheduler = ExponentialLR(optimizer, gamma=hyperparams["gamma"])

    os.makedirs(outDir / "states", exist_ok=True)

    # taken from https://docs.pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    checkpoint: Checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpointDir:
            dataPath = f"{checkpointDir}/data.pkl"
            with open(dataPath, "rb") as f:
                checkpointState = pickle.load(f)
            starting_epoch = checkpointState["epoch"]
            model.load_state_dict(checkpointState["model_state_dict"])
            optimizer.load_state_dict(checkpointState["optimizer_state_dict"])
            losses = checkpointState["losses"]
            valLosses = checkpointState["validation_losses"]
    else:
        resetAllWeights(model)
        starting_epoch = 0
        losses = []
        valLosses = []

    rng = torch.Generator().manual_seed(56328417)
    trainSet, valSet = random_split(trainingDataset, (0.8, 0.2), rng)

    trainingDataLoader = DataLoader(
        trainSet, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=0
    )
    valDataLoader = DataLoader(
        valSet, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=0
    )

    # TODO THE REASON TESTING < TRAINING IS THAT ITS ONLY TAKEN AT THE BEST ITERATION
    # training loop, based on the one provided by pytorch
    # TODO split validation for
    for epoch in range(starting_epoch, epochs):
        printANSI(f"-=-=-=-=- EPOCH {epoch + 1} -=-=-=-=-", "bold", "yellow")

        print("training...")
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        avg_loss = trainOnce(
            model, trainingDataLoader, criterion, optimizer, scheduler, device
        )
        losses.append(avg_loss)

        runningValLoss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        print("validating...")
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for valInputs, valLabels in tqdm(iter(valDataLoader)):
                valInputs = valInputs.to(device)
                valLabels = valLabels.to(device)

                valOutputs = model(valInputs)
                valLoss = criterion(valOutputs, valLabels)
                runningValLoss += valLoss

        avgValLoss = runningValLoss / (len(valDataLoader) + 1)
        valLosses.append(avgValLoss)

        # report losses
        print(f"Training loss (MSE): {avg_loss}")
        print(f"Validation loss (MSE): {avgValLoss}")

        # finally, save the model state
        checkpointData = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "validation_losses": valLosses,
        }

        checkpointDir = outDir / f"epoch_{epoch}" / "data.pkl"

        with open(checkpointDir) as f:
            pickle.dump(checkpointData, f)

        checkpoint = Checkpoint.from_directory(checkpointDir)
        ray.train.report({"loss": avgValLoss}, checkpoint=checkpoint)

    # retrieve losses from gpu
    valLosses = [loss.cpu() for loss in valLosses]

    return losses, valLosses


# endregion


# region Main Execution
def main(argv=None):
    args = parser.parse_args(argv)
    # currentTime = datetime.datetime.now()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator().manual_seed(1)

    printANSI(f"running torch on {device}", "bold", "bright_magenta")

    dataset = YTDataset(args.imageDir)
    trainingSet, testingSet = random_split(dataset, (0.85, 0.15), generator=rng)

    printBox(
        f"dataset loaded with {len(dataset)} entries ({len(trainingSet)} training, {len(testingSet)} testing)",  # noqa: E501
        "bold",
        "bright_magenta",
    )

    # testingDataLoader = DataLoader(
    #     testingSet, batch_size=args.batch_size, shuffle=True, num_workers=0
    # )

    # TODO: add flag for displaying these things
    # gridSize = (4, 6)
    # index = np.random.randint(0, len(dataset) - np.prod(gridSize))
    # showGrid(gridSize, dataset[index:index+np.prod(gridSize)][0])

    model = ThumbnailModel(*dataset.imageSize).to(device)

    printBox(
        f"model loaded with {getNumParams(model)} parameters",
        "bold",
        "bright_magenta",
    )

    hyperparams = {
        "lr": ray.tune.loguniform(1e-5, 1e-2),
        "gamma": ray.tune.uniform(0.75, 1),
        "batch_size": ray.tune.choice(range(8, 28, 4)),
    }
    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=args.epochs, grace_period=1, reduction_factor=2
    )
    result = ray.tune.run(
        partial(
            train, trainingDataset=trainingSet, epochs=args.epochs, outDir=args.outDir
        ),
        resources_per_trial={"cpu": 2, "gpu": 2},
        config=hyperparams,
        num_samples=10,
        scheduler=scheduler,
    )

    print(result)

    # losses, v_losses = train(
    #     model,
    #     args.epochs,
    #     trainingSet,
    #     hyperparams,
    #     outDir=args.outDir,
    # )

    # with open(
    #     f"{args.outDir}/losses_{currentTime.isoformat()}.csv",
    #     "w",
    # ) as f:
    #     fileBody = "loss,val loss\n"
    #     for i in range(len(losses)):
    #         fileBody += f"{losses[i]},{v_losses[i]}\n"
    #     f.write(fileBody)

    # fig = plt.figure()

    # plt.plot(range(len(v_losses)), v_losses, label="validation", color="#fa96c8")
    # plt.plot(range(len(losses)), losses, label="training", color="#6496fa")
    # plt.legend()

    # fig.show()
    # fig.savefig(f"{args.outDir}/losses.png")


# endregion


# region Entry Point
if __name__ == "__main__":
    main()
# endregion
