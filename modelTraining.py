import os
from tqdm import tqdm
import datetime
import re
import argparse

import torch
from torch.utils.data import DataLoader, random_split, default_collate
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import matplotlib.pyplot as plt

from modelDataset import YTDataset
from model import ThumbnailModel
from printHelper import getANSI, resetANSI

# region arguments
parser = argparse.ArgumentParser(
    prog="python3 modelTraining.py",
    description="Trains a neural network on YouTube data",
    epilog="Copyright Liam Hillery, 2025",
)

parser.add_argument_group("File I/O")
parser.add_argument("-i", "--imageDir", default="./output/thumbnails")
parser.add_argument("-o", "--outDir", default="./output")
parser.add_argument("-s", "--starting-weights", default=None, type=str)

parser.add_argument_group("Training options")
parser.add_argument("-e", "--epochs", default=5, type=int)
parser.add_argument("-b", "--batch-size", default=20, type=int)
parser.add_argument(
    "-v", "--validation-only", action="store_true"
)  # TODO: not yet implemented
# endregion


# region showGrid
def showGrid(size, images, text=None, filename=None):
    figure, axs = plt.subplots(nrows=size[0], ncols=size[1])
    figure.tight_layout()
    figure.set_size_inches(size[1] * 3, size[0] * 3)

    for i in range(len(axs)):
        for j in range(len(axs[i])):
            axs[i][j].imshow(images[i * len(axs[i]) + j])
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            if text:
                axs[i][j].set_title(text[i * len(axs[i]) + j])

    plt.show()

    if filename:
        plt.savefig(filename, dpi=500)


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


# region getNumParams
def getNumParams(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# endregion


# region Main Execution
def main(argv=None):
    args = parser.parse_args(argv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ansi = getANSI("bold", "bright_magenta")
    print(f"{ansi}running torch on {device}{resetANSI()}")

    dataset = YTDataset(args.imageDir)

    rng = torch.Generator().manual_seed(1)
    trainingSet, testingSet = random_split(dataset, (0.85, 0.15), generator=rng)

    datasetLoadedStatement = f"dataset loaded with {len(dataset)} entries ({len(trainingSet)} training, {len(testingSet)} testing)"  # noqa: E501

    print(getANSI("bold", "bright_magenta"))
    print("+-" + "".join(["-" for c in datasetLoadedStatement]) + "-+")
    print("| " + "".join([" " for c in datasetLoadedStatement]) + " |")
    print("| " + datasetLoadedStatement + " |")
    print("| " + "".join([" " for c in datasetLoadedStatement]) + " |")
    print("+-" + "".join(["-" for c in datasetLoadedStatement]) + "-+", resetANSI())

    # print(dataset[1000])
    subset = dataset[:100]
    print(dataset[0][0].shape)
    print(subset[0].shape)

    # showGrid((4, 6), subset[:9][0])

    trainingDataLoader = DataLoader(
        trainingSet, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    testingDataLoader = DataLoader(
        testingSet, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    model = ThumbnailModel(*dataset.imageSize).to(device)
    print(getNumParams(model))

    output = model.forward(
        default_collate([testingSet[i][0].to(device) for i in range(2)])
    )
    print(output.size())
    print(output)
    # plt.imshow(npImage(output[0,:1].cpu()))
    # print(output[0][0].cpu().detach().numpy())

    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()

    # optimizer = SGD(model.parameters(), lr=5e-6, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=1e-4)

    scheduler = ExponentialLR(optimizer, gamma=0.9)

    losses, v_losses = [], []

    os.makedirs(f"{args.outDir}/states", exist_ok=True)

    # set to True to resume from a specified epoch
    start = 0

    if args.starting_weights:
        # attempt to derive epoch number from filename
        match = re.match(r".*epoch_(\d+)\.pt", args.starting_weights)

        if re.match(r"", args.starting_weights):
            start = match.group(1)

        # load weights
        model.load_state_dict(
            torch.load(
                f"{args.starting_weights}epoch_{start+1}.pt", map_location=device
            )
        )

    # TODO THE REASON TESTING < TRAINING IS THAT ITS ONLY TAKEN AT THE BEST ITERATION
    # training loop, based on the one provided by pytorch
    for epoch in range(start, args.epochs):
        ansi = getANSI("bold", "yellow")
        print(f"{ansi}-=-=-=-=- EPOCH {epoch + 1} -=-=-=-=-{resetANSI()}")

        print("training...")
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        avg_loss = trainOnce(
            model, trainingDataLoader, criterion, optimizer, scheduler, device
        )
        losses.append(avg_loss)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        print("validating...")
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for vinputs, vlabels in tqdm(iter(testingDataLoader)):
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)

                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (len(testingDataLoader) + 1)
        v_losses.append(avg_vloss)

        # report losses
        print(f"Training loss (MSE): {avg_loss}")
        print(f"Testing loss (MSE): {avg_vloss}")

        # finally, save the model params for future reference
        torch.save(model.state_dict(), f"{args.outDir}/states/epoch_{epoch + 1}.pt")

    # retrieve losses from gpu and plot them
    v_losses = [loss.cpu() for loss in v_losses]

    currentTime = datetime.datetime.now()

    with open(
        f"{args.outDir}/losses_{currentTime.isoformat()}.csv",
        "w",
    ) as f:
        fileBody = "loss,val loss\n"
        for i in range(len(losses)):
            fileBody += f"{losses[i]},{v_losses[i]}\n"
        f.write(fileBody)

    plt.figure()

    plt.plot(range(len(v_losses)), v_losses, label="validation", color="#fa96c8")
    plt.plot(range(len(losses)), losses, label="training", color="#6496fa")
    plt.legend()

    plt.show()


# endregion


# region Entry Point
if __name__ == "__main__":
    main()
# endregion
