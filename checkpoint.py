import os
import re
import pickle
from pathlib import Path


def getLatestEpoch(checkpointDir: Path) -> int:
    dirContents = "\n".join(os.listdir(checkpointDir))

    results: list[re.Match] = re.findall(r"^checkpoint_epoch_{\d+}\.pkl$", dirContents)

    if not results:
        return None

    greatestEpoch = -1

    for match in results:
        greatestEpoch = max(greatestEpoch, int(match.group(1)))

    return greatestEpoch


def loadCheckpoint(checkpointFile: Path):
    with open(checkpointFile, "rb") as f:
        return pickle.load(f)


def loadLatestCheckpoint(checkpointDir: Path):
    epoch: int = getLatestEpoch(checkpointDir)

    if epoch is None:
        return None

    return loadCheckpoint(checkpointDir / f"checkpoint_epoch_{epoch}.pkl")


def saveCheckpoint(checkpointFile: Path, data: dict):
    with open(checkpointFile, "rb") as f:
        return pickle.dump(data, f)


def saveLatestCheckpoint(checkpointDir: Path, data: dict, epoch: int = None):
    epoch: int = epoch or getLatestEpoch(checkpointDir) or 1

    saveCheckpoint(checkpointDir / f"checkpoint_epoch_{epoch}.pkl", data)
