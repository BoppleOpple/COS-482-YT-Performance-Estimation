import os
import psycopg2
import datetime
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

import PIL.Image

import dataResourceDownload
from ansi import *

class YTDataset (Dataset):
	images = dict()
	secondsDifference = np.vectorize(lambda duration: duration.total_seconds())
	imageSize = (640, 480)

	def __init__(self, imageDir = "./output/thumbnails"):
		dbConnection: psycopg2.extensions.connection = psycopg2.connect(
			host=os.environ["SQL_HOST"],
			port=os.environ["SQL_PORT"],
			dbname=os.environ["SQL_DBNAME"],
			user=os.environ["SQL_USER"],
			password=os.environ["SQL_PASSWORD"]
		)

		print()
		print(f"{getANSI("bold", "bright_blue")}updating thumbnails...{resetANSI()}")
		dataResourceDownload.main()

		# retrieve all video IDS
		videoIDs = None
		with dbConnection.cursor() as cursor:
			cursor.execute("SELECT id FROM videos")
			result = cursor.fetchall()
			videoIDs = np.array(result)[:,0]
		
		invalidThumbnails = []

		brokenThumbnailImage: PIL.ImageFile.ImageFile = PIL.Image.open("res/brokenThumbnail.jpg")

		print()
		print(f"{getANSI("bold", "bright_blue")}loading thumbnails...{resetANSI()}")
		# find thumbnails
		for vid in tqdm(videoIDs):
			if vid not in self.images:
				imagePath = f"{imageDir}/{vid}.jpg"
				image: PIL.ImageFile.ImageFile = PIL.Image.open(imagePath)

				if brokenThumbnailImage == image:
					invalidThumbnails.append(vid)
				else:
					self.images[vid] = np.array(image.resize(self.imageSize))

		# Download data from the database
		with dbConnection.cursor() as cursor:
			cursor.execute("\
				SELECT access_information.video_id, posted_time, query_time, subscribers, access_information.views, likes, comments \
				FROM access_information, created_by, channels, videos \
				WHERE ( \
					access_information.video_id = created_by.video_id \
					AND created_by.channel_id = channels.id \
					AND access_information.video_id = videos.id \
				);\
			")
			result = cursor.fetchall()
			print()
			print(f"{getANSI("bold", "bright_blue")}filtering data...{resetANSI()}")
			self.data = np.array(list(filter(lambda data: data[0] not in invalidThumbnails, tqdm(result))))
		print()
		print(self.data.shape)
		print(self.data[:5])

		# load the thumbnails into a dictionary
		
		dbConnection.close()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		row = self.data[idx]

		if len(row.shape) > 1:
			inputImages = np.apply_along_axis(lambda r: self.images[r[0]], 1, row)

			inputData = np.array((
				self.secondsDifference(row[:,2] - row[:,1]), # time since posting
				row[:,3]
			))
			outputData = row[:,-3:]
			
			return inputImages, inputData, outputData
		else:
			inputImage = self.images[row[0]]

			inputData = np.array((
				int((row[2] - row[1]).total_seconds()), # time since posting
				int(row[3])
			))
			
			outputData = row[-3:]

			return inputImage, inputData, outputData


def show_grid(size, images, text=None, filename=None):
	figure, axs = plt.subplots(nrows=size[0], ncols=size[1])
	# figure.tight_layout()
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


if __name__ == "__main__":
	load_dotenv()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print(f"{getANSI("bold", "bright_magenta")}running torch on {device}{resetANSI()}")

	dataset = YTDataset()

	datasetLoadedStatement = f"dataset loaded with {len(dataset)} entries"

	print(getANSI("bold", "bright_magenta"))
	print(f"+-{"".join(["-" for c in datasetLoadedStatement])}-+")
	print(f"| {"".join([" " for c in datasetLoadedStatement])} |")
	print(f"| {datasetLoadedStatement} |")
	print(f"| {"".join([" " for c in datasetLoadedStatement])} |")
	print(f"+-{"".join(["-" for c in datasetLoadedStatement])}-+{resetANSI()}")

	# print()
	# print(dataset[0])

	# print()
	# print(dataset[0][0])

	# print()
	# print(dataset[:9][0])

	subset = dataset[:1000]

	show_grid((3, 3), subset[:9][0])