import os
import argparse
from pathlib import Path
import pickle
import datetime

import torch
from torch.utils.data import random_split

from modelDataset import YTDataset
from printHelpers import printANSI
from tuner import tune, LinearRange, Selection

# region arguments
parser = argparse.ArgumentParser(
    prog="python3 run.py",
    description="Tunes and Trains a neural network on YouTube data",
    epilog="Copyright Liam Hillery, 2025",
)

parser.add_argument_group("File I/O")
parser.add_argument("-i", "--imageDir", default="./output/thumbnails", type=Path)
parser.add_argument("-o", "--outDir", default="./output", type=Path)
parser.add_argument("-s", "--sessionName", default=None, type=str)

parser.add_argument_group("Training options")
parser.add_argument("-e", "--epochs", default=5, type=int)
parser.add_argument("--validation_epochs", default=None, type=int)
# endregion

IMAGE_SIZE = (640, 360)
VOCAB_DIMS = 300


# region Main Execution
def main(argv=None):
    args = parser.parse_args(argv)
    currentTime = datetime.datetime.now()

    sessionName = args.sessionName or f"session_{currentTime.isoformat()}"

    sessionPath: Path = args.outDir / sessionName
    hyperparameterPath: Path = sessionPath / "tuningParams.pkl"
    dbInfoPath: Path = sessionPath / "dataInfo.json"

    os.makedirs(sessionPath, exist_ok=True)

    if hyperparameterPath.exists():
        printANSI(
            "Loading hyperparameters from session dir (you have been warned)",
            "bold",
            "cyan",
        )
        with open(hyperparameterPath, "rb") as f:
            hyperparams = pickle.load(f)
    else:
        hyperparams = {
            "rnn_hidden_layer_size": Selection([int(i) for i in range(128, 513, 64)]),
            "lr": LinearRange(5e-6, 1e-5),
            "gamma": LinearRange(0.5, 0.75),
            "batch_size": Selection([int(i) for i in reversed(range(2, 12, 2))]),
        }

    if dbInfoPath.exists():
        with open(dbInfoPath, "rb") as f:
            dbInfo = pickle.load(f)
    else:
        dbInfo = {
            "image_size": IMAGE_SIZE,
            "vocab_dims": VOCAB_DIMS,
            "time": currentTime,
        }
        with open(dbInfoPath, "wb") as f:
            pickle.dump(dbInfo, f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator().manual_seed(1)

    printANSI(f"running torch on {device}", "bold", "bright_magenta")

    dataset = YTDataset(
        args.imageDir,
        imageSize=dbInfo["image_size"],
        vocabDims=dbInfo["vocab_dims"],
        endDateTime=dbInfo["time"],
    )

    trainingSet, valSet, testingSet = random_split(
        dataset, (0.75, 0.10, 0.15), generator=rng
    )

    # testingDataLoader = DataLoader(
    #     testingSet, batch_size=args.batch_size, shuffle=True, num_workers=0
    # )

    # TODO: add flag for displaying these things
    # gridSize = (4, 6)
    # index = np.random.randint(0, len(dataset) - np.prod(gridSize))
    # showGrid(gridSize, dataset[index:index+np.prod(gridSize)][0])

    tune(
        hyperparams,
        trainingSet,
        valSet,
        sessionPath,
        epochs=args.epochs,
        parameterSamples=5,
        collate_fn=dataset.collate_fn,
    )


# endregion


# region Entry Point
if __name__ == "__main__":
    main()
# endregion
