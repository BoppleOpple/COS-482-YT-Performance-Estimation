import os
import psycopg2
import datetime
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

import torch
from torch.utils.data import Dataset

import PIL.Image

import dataResourceDownload
from ansi import *

class YTDataset (Dataset):
	images = dict()

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
					self.images[vid] = np.array(image)

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

		inputImage = self.images[row[0]]

		print()
		print(inputImage.shape)

		inputData = np.array((
			int((row[2] - row[1]).total_seconds()), # time since posting
			int(row[3])
		))
		
		outputData = np.array((row[-3], row[-2], row[-1]))

		return inputImage, inputData, outputData

if __name__ == "__main__":
	load_dotenv()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print(f"{getANSI("bold", "magenta")}running torch on {device}{resetANSI()}")

	dataset = YTDataset()

	print()
	print(f"{getANSI("bold", "magenta")}dataset loaded with {len(dataset)} entries{resetANSI()}")

	print(dataset[0])