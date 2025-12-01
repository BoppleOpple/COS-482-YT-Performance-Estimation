import argparse
from pathlib import Path
import datetime

import torch
from torch.utils.data import random_split

from modelDataset import YTDataset
from printHelpers import printANSI
from tuner import tune, LinearRange, NormalRange, Selection

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


# region Main Execution
def main(argv=None):
    args = parser.parse_args(argv)
    currentTime = datetime.datetime.now()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator().manual_seed(1)

    printANSI(f"running torch on {device}", "bold", "bright_magenta")

    dataset = YTDataset(args.imageDir, IMAGE_SIZE)
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

    hyperparams = {
        "lr": NormalRange(1e-5, 1e-2),
        "gamma": LinearRange(0.75, 1),
        "batch_size": Selection([int(i) for i in range(4, 24, 4)]),
    }

    session_name = args.sessionName or f"session_{currentTime.isoformat()}"

    tune(
        hyperparams,
        trainingSet,
        valSet,
        args.outDir / session_name,
        epochs=args.epochs,
        parameterSamples=5,
    )


# endregion


# region Entry Point
if __name__ == "__main__":
    main()
# endregion
