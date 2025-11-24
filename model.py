import torch
import torchvision


class ThumbnailModel(torch.nn.Module):
    def __init__(self, w, h):
        super().__init__()
        # mobilenet v2
        pretrainedMobileNet = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2")

        # extract the appropriate layers from the pretrained mobilenet model
        self.featureExtraction1 = pretrainedMobileNet.features[0:2]  # 16 channels
        self.featureExtraction2 = pretrainedMobileNet.features[2:4]  # 24 channels
        self.featureExtraction3 = pretrainedMobileNet.features[4:7]  # 32 channels
        self.featureExtraction4 = pretrainedMobileNet.features[7:11]  # 64 channels
        self.featureExtraction5 = pretrainedMobileNet.features[11:17]  # 160 channels

        self.flatten = torch.nn.Flatten()

        print(f"latents: ({ w//16 }, { h//16 })")
        self.fc1 = torch.nn.Linear(w // 32 * h // 32 * 160, 160)
        self.fc2 = torch.nn.Linear(160, 16)
        self.fc3 = torch.nn.Linear(16, 3)

        self.float()

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

        layer6 = self.fc1(layer5)
        # print(f"layer6: {layer6.shape}")
        layer7 = self.fc2(layer6)
        # print(f"layer7: {layer7.shape}")
        layer8 = self.fc3(layer7)
        # print(f"layer8: {layer8.shape}")

        return layer8


if __name__ == "__main__":
    exit(0)
