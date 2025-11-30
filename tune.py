import os
import json
import datetime
import numpy as np
from scipy.stats import norm
from pathlib import Path

from torch.utils.data import Dataset

from modelTraining import train
from printHelpers import printANSI, printBox


# region normalQuantile
# implementation translated from r source
@np.vectorize()
def normalQuantile(p: float):
    return norm.ppf(p)


# endregion


# region TuneParam
class TuneParam:
    def select(self, n: int = 5) -> np.ndarray:
        raise NotImplementedError(
            "TuneParam is a base class, try LinerRange or NormalRange"
        )


# endregion


# region LinearRange
class LinearRange(TuneParam):
    def __init__(self, min_range: float, max_range: float):
        self.min = min_range
        self.max = max_range

    def select(self, n: int = 5) -> np.ndarray:
        if n <= 0:
            return np.array()
        elif n == 1:
            return np.array([(self.max - self.min) / 2])

        step = (self.max - self.min) / (n - 1)
        return np.arange(self.min, self.max + step / 2, step)


# endregion


# region NormalRange
class NormalRange(TuneParam):
    def __init__(self, min_range: float, max_range: float, p_value: float = 0.95):
        self.min = min_range
        self.max = max_range

        mean = (self.min + self.max) / 2

        self.sd = (self.max - mean) / norm.ppf(0.5 + p_value / 2)

    def select(self, n: int = 5) -> np.ndarray:
        mean = (self.min + self.max) / 2

        if n <= 0:
            return np.array()
        elif n == 1:
            return np.array([mean])
        elif n == 2:
            return np.array([self.min, self.max])

        p_step = 1 / (n - 1)

        return np.concat(
            [
                [self.min],
                mean + normalQuantile(np.arange(0, 1, p_step)[1:]) * self.sd,
                [self.max],
            ]
        )


# endregion


# region ConstantRange
class ConstantRange(TuneParam):
    def __init__(self, value):
        self.value = value

    def select(self, n: int = 5) -> np.ndarray:
        return np.array([self.value])


# endregion


# region tune
def tune(
    hyperparams: dict[str, TuneParam],
    trainingSet: Dataset,
    validationSet: Dataset,
    outDir: Path,
    epochs: int = 5,
    parameterSamples: int = 5,
    method: str = "cascade",
):
    currentTime = datetime.datetime.now()

    bestParams = dict()
    if method == "cascade":
        for param, value in hyperparams.items():
            bestParams[param] = value.select(1)

        trialID = 1
        for param in hyperparams:
            printBox(f"optimizing '{param}'", "bold", "bright_yellow")
            paramOptions = hyperparams[param].select(parameterSamples)
            paramLosses: np.ndarray = np.full((len(paramOptions), 2, epochs), np.inf)
            for i, value in enumerate(paramOptions):
                printANSI(f"beginning trial {trialID}", "bold", "bright_blue")
                trialDir = outDir / f"trial_{currentTime.isoformat()}"

                trialParams = bestParams.copy()
                trialParams[param] = value

                os.makedirs(trialDir, exist_ok=True)
                with open(trialDir / "params.json", "w") as jsonFile:
                    json.dump(trialParams, jsonFile)

                testLosses, valLosses = train(
                    trialParams,
                    epochs,
                    trainingSet,
                    validationSet,
                    outDir / f"trial_{currentTime.isoformat()}",
                )

                paramLosses[i, 0] = testLosses
                paramLosses[i, 1] = valLosses
                trialID += 1

            paramValidationLosses = paramLosses[:, 1, :]

            bestValue = paramOptions[np.argmin(np.min(paramValidationLosses, 1))]
            printANSI(
                f"best value found for '{param}': {bestValue}", "bold", "bright_green"
            )

            bestParams[param] = bestValue

        return bestParams


# endregion

# region Entry Point
if __name__ == "__main__":
    lRange = LinearRange(0, 5)
    nRange = NormalRange(-1, 1)

    lSelection = lRange.select(5)
    print(f"linear range (0, 5) with 5 elements: {lSelection}")

    nSelection = nRange.select(5)
    print(f"normal range (-1, 1) with 5 elements: {nSelection}")

    # hyperparams = {
    #     "param1": lRange,
    #     "param2": nRange
    # }

    # print(tune(hyperparams, None, None, None, epochs=10))
# endregion
