import math
import torch
import torchvision


# region ThumbnailModel
class ThumbnailModel(torch.nn.Module):

    # region __init__
    def __init__(self, w, h):
        super().__init__()
        self.nonLinear = torch.nn.ReLU()

        # mobilenet v2
        pretrainedMobileNet = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2")

        # extract the appropriate layers from the pretrained mobilenet model
        self.featureExtraction1 = pretrainedMobileNet.features[0:2]  # 16 channels
        self.featureExtraction2 = pretrainedMobileNet.features[2:4]  # 24 channels
        self.featureExtraction3 = pretrainedMobileNet.features[4:7]  # 32 channels
        self.featureExtraction4 = pretrainedMobileNet.features[7:11]  # 64 channels
        self.featureExtraction5 = pretrainedMobileNet.features[11:17]  # 160 channels

        self.flatten = torch.nn.Flatten()

        print(f"latents: ({ math.ceil(w/32) }, { math.ceil(h/32) }, 160)")
        self.fc1 = torch.nn.Linear(math.ceil(w / 32) * math.ceil(h / 32) * 160, 160)
        self.fc2 = torch.nn.Linear(160, 16)
        self.fc3 = torch.nn.Linear(16, 3)

        self.float()

    # endregion

    # region forward
    def forward(self, x: torch.Tensor):
        layer0 = self.featureExtraction1(x)
        # print(f"layer0: {layer0.shape}")
        layer1 = self.featureExtraction2(layer0)
        # print(f"layer1: {layer1.shape}")
        layer2 = self.featureExtraction3(layer1)
        # print(f"layer2: {layer2.shape}")
        layer3 = self.featureExtraction4(layer2)
        # print(f"layer3: {layer3.shape}")
        layer4 = self.featureExtraction5(layer3)
        # print(f"layer4: {layer4.shape}")

        layer5 = self.flatten(layer4)
        # print(f"layer5: {layer5.shape}")

        layer6 = self.nonLinear(self.fc1(layer5))
        # print(f"layer6: {layer6.shape}")
        layer7 = self.nonLinear(self.fc2(layer6))
        # print(f"layer7: {layer7.shape}")
        layer8 = self.fc3(layer7)
        # print(f"layer8: {layer8.shape}")

        return layer8

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


# region resetAllWeights
def resetAllWeights(model: torch.nn.Module) -> None:
    # source: https://discuss.pytorch.org/t/reset-model-weights/19180/7

    @torch.no_grad()
    def weightReset(m: torch.nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule:
    # see https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weightReset)


# endregion
