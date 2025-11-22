import os
from tqdm import tqdm
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader, random_split, default_collate
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR

import matplotlib.pyplot as plt

from modelDataset import *
from ansi import *

class ThumbnailModel (torch.nn.Module):
	def __init__(self, w, h):
		super().__init__()
		# mobilenet v2
		pretrainedMobileNet = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V2')

		# extract the appropriate layers from the pretrained mobilenet model
		self.featureExtraction1 = pretrainedMobileNet.features[0:2]   # 16 channels
		self.featureExtraction2 = pretrainedMobileNet.features[2:4]   # 24 channels
		self.featureExtraction3 = pretrainedMobileNet.features[4:7]   # 32 channels
		self.featureExtraction4 = pretrainedMobileNet.features[7:11]  # 64 channels
		self.featureExtraction5 = pretrainedMobileNet.features[11:17] # 160 channels

		self.flatten = torch.nn.Flatten()

		print(f"latents: ({ w//16 }, { h//16 })")
		self.fc1 = torch.nn.Linear(w//32 * h//32 * 160, 160)
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


def show_grid(size, images, text=None, filename=None):
	figure, axs = plt.subplots(nrows=size[0], ncols=size[1])
	figure.tight_layout()
	figure.set_size_inches(size[1]*3, size[0]*3)

	for i in range(len(axs)):
		for j in range(len(axs[i])):
			axs[i][j].imshow(images[i*len(axs[i]) + j])
			axs[i][j].set_xticks([])
			axs[i][j].set_yticks([])
			if text:
				axs[i][j].set_title(text[i*len(axs[i]) + j])
	
	plt.show()
	
	if filename:
		plt.savefig(filename, dpi=500)

# epoch function based on the default provided by pytorch
def train_one_epoch(model, dataLoader, optimizer, scheduler):
	criterion = torch.nn.MSELoss()
	# criterion = torch.nn.L1Loss()

	running_loss = 0.

	batch = 1

	for inputs, groundTruth in tqdm(iter(dataLoader)):
		inputs = inputs.to(device)
		groundTruth = groundTruth.to(device)

		# Zero your gradients for every batch!
		optimizer.zero_grad()

		# Make predictions for this batch
		output = model(inputs)

		# print(output.dtype)
		# print(groundTruth.dtype)
		
		# Compute the loss and its gradients
		loss = criterion(output, groundTruth)
		loss.backward()
		print(loss)

		# Adjust learning weights
		optimizer.step()

		# Gather data and report
		running_loss += loss.item()
		batch += 1
		if batch % 20 == 0: print(f'{getANSI("bold", "yellow")}batch {batch} / {len(dataLoader)} | training loss: {loss.item()}{resetANSI()}')
	
	scheduler.step()

	return running_loss / len(dataLoader)

def getNumParams(model):
	pp=0
	for p in list(model.parameters()):
		nn=1
		for s in list(p.size()):
			nn = nn*s
		pp += nn
	return pp



if __name__ == "__main__":
	load_dotenv()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print(f"{getANSI("bold", "bright_magenta")}running torch on {device}{resetANSI()}")

	dataset = YTDataset()

	rng = torch.Generator().manual_seed(1)
	trainingSet, testingSet = random_split(dataset, (0.85, 0.15), generator=rng)

	datasetLoadedStatement = f"dataset loaded with {len(dataset)} entries ({len(trainingSet)} training, {len(testingSet)} testing)"

	print(getANSI("bold", "bright_magenta"))
	print(f"+-{"".join(["-" for c in datasetLoadedStatement])}-+")
	print(f"| {"".join([" " for c in datasetLoadedStatement])} |")
	print(f"| {datasetLoadedStatement} |")
	print(f"| {"".join([" " for c in datasetLoadedStatement])} |")
	print(f"+-{"".join(["-" for c in datasetLoadedStatement])}-+{resetANSI()}")

	# print(dataset[1000])
	subset = dataset[:100]
	print(dataset[0][0].shape)
	print(subset[0].shape)

	# show_grid((4, 6), subset[:9][0])

	trainingDataLoader = DataLoader(trainingSet, batch_size=8, shuffle=True, num_workers=0)
	testingDataLoader = DataLoader(testingSet, batch_size=8, shuffle=True, num_workers=0)

	model = ThumbnailModel(*dataset.imageSize)
	print(getNumParams(model))

	output = model.forward(default_collate([testingSet[i][0].to(device) for i in range(2)]))
	print(output.size())
	print(output)
	# plt.imshow(npImage(output[0,:1].cpu()))
	# print(output[0][0].cpu().detach().numpy())


	optimizer = SGD(model.parameters(), lr=5e-6, momentum=0.9)
	# optimizer = Adam(model.parameters(), lr=1e-4)

	scheduler = ExponentialLR(optimizer, gamma=0.9)

	print(train_one_epoch(model, trainingDataLoader, optimizer, scheduler))
	print(train_one_epoch(model, trainingDataLoader, optimizer, scheduler))
