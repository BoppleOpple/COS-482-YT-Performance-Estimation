import math
import torch
import torchvision

# import torchtext

from printHelpers import printBox


# region ThumbnailModel
class ThumbnailModel(torch.nn.Module):
    def __init__(self, w, h):
        super().__init__()
        self.nonLinear = torch.nn.ReLU()

        # mobilenet v2
        pretrainedMobileNet = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2")

        # extract the appropriate layers from the pretrained mobilenet model
        self.featureExtraction = pretrainedMobileNet.features[:17]  # 160 channels

        self.flatten = torch.nn.Flatten()

        print(f"latents: ({ math.ceil(w/32) }, { math.ceil(h/32) }, 160)")
        self.fc1 = torch.nn.Linear(math.ceil(w / 32) * math.ceil(h / 32) * 160, 160)
        self.fc2 = torch.nn.Linear(self.fc1.out_features, 16)
        self.fc3 = torch.nn.Linear(self.fc2.out_features, 3)

        self.features = torch.nn.Sequential(
            self.featureExtraction,
            self.flatten,
            self.fc1,
            self.nonLinear,
            self.fc2,
            self.nonLinear,
            self.fc3,
        )

        self.float()

    # region ThumbnailModel.forward
    def forward(self, x: torch.Tensor):
        # layer0 = self.featureExtraction1(x)
        # # print(f"layer0: {layer0.shape}")
        # layer1 = self.featureExtraction2(layer0)
        # # print(f"layer1: {layer1.shape}")
        # layer2 = self.featureExtraction3(layer1)
        # # print(f"layer2: {layer2.shape}")
        # layer3 = self.featureExtraction4(layer2)
        # # print(f"layer3: {layer3.shape}")
        # layer4 = self.featureExtraction5(layer3)
        # # print(f"layer4: {layer4.shape}")

        # layer5 = self.flatten(layer4)
        # # print(f"layer5: {layer5.shape}")

        # layer6 = self.nonLinear(self.fc1(layer5))
        # # print(f"layer6: {layer6.shape}")
        # layer7 = self.nonLinear(self.fc2(layer6))
        # # print(f"layer7: {layer7.shape}")
        # layer8 = self.fc3(layer7)
        # # print(f"layer8: {layer8.shape}")

        return self.features(x)

    # endregion


# endregion


# region YTModel
class YTModel(torch.nn.Module):
    def __init__(self, w, h, vocabularyDim, hiddenSize, extraParams=2):
        super().__init__()
        self.nonLinear = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()

        self.text_rnn = torch.nn.RNN(
            input_size=vocabularyDim,
            hidden_size=hiddenSize,
            num_layers=2,
            nonlinearity="relu",
            batch_first=True,
        )

        thumbnailModel = ThumbnailModel(w, h)
        self.thumbnailEncoder = thumbnailModel.features[:-1]  # 16 channels

        self.fc1 = torch.nn.Linear(
            16 + self.text_rnn.num_layers * hiddenSize + extraParams, 64
        )
        self.fc2 = torch.nn.Linear(self.fc1.out_features, 32)
        self.fc3 = torch.nn.Linear(self.fc2.out_features, 16)
        self.fc4 = torch.nn.Linear(self.fc3.out_features, 3)

        self.linearLayers = torch.nn.Sequential(
            self.fc1,
            self.nonLinear,
            self.fc2,
            self.nonLinear,
            self.fc3,
            self.nonLinear,
            self.fc4,
        )

        self.float()

        printBox(
            f"model loaded with {getNumParams(self)} parameters",
            "bold",
            "bright_magenta",
        )

    # region YTModel.forward
    def forward(self, thumbnails, encodedTitles, x):
        thumbnailFeatures = self.thumbnailEncoder(thumbnails)
        _, hiddenOutput = self.text_rnn(encodedTitles)

        titleFeatures = torch.swapaxes(hiddenOutput, 0, 1)

        features = torch.cat((thumbnailFeatures, self.flatten(titleFeatures), x), dim=1)

        return self.linearLayers(features)

    # endregion


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
