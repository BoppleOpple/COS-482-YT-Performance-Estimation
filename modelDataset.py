import os
import psycopg2
from threading import Thread
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

import torch
from torch.utils.data import Dataset

import PIL.Image

import dataResourceDownload
from printHelper import getANSI, resetANSI

brokenThumbnailImage: PIL.ImageFile.ImageFile = PIL.Image.open(
    "res/brokenThumbnail.jpg"
)


# region Helpers
def npToMpl(image: np.ndarray):
    return np.swapaxes(np.swapaxes(image, 1, 2), 0, 2)


def mplToNp(image: np.ndarray):
    return np.swapaxes(np.swapaxes(image, 0, 2), 1, 2)


def validThumbnail(imageDir: str, vid: str):
    imagePath = f"{imageDir}/{vid}.jpg"
    return not brokenThumbnailImage == PIL.Image.open(imagePath)


# endregion


# region YTDataset
class YTDataset(Dataset):
    images = dict()
    secondsDifference = np.vectorize(lambda duration: duration.total_seconds())
    imageSize = (640, 480)

    # region __init__
    def __init__(self, imageDir="./output/thumbnails"):
        load_dotenv()

        self.imageDir = imageDir

        dbConnection: psycopg2.extensions.connection = psycopg2.connect(
            host=os.environ["SQL_HOST"],
            port=os.environ["SQL_PORT"],
            dbname=os.environ["SQL_DBNAME"],
            user=os.environ["SQL_USER"],
            password=os.environ["SQL_PASSWORD"],
        )

        ansi = getANSI("bold", "bright_blue")
        print()
        print(f"{ansi}updating thumbnails...{resetANSI()}")
        dataResourceDownload.main()

        # Download data from the database
        with dbConnection.cursor() as cursor:
            cursor.execute(
                "\
                SELECT \
                    access_information.video_id, \
                    posted_time, \
                    query_time, \
                    subscribers, \
                    access_information.views, \
                    likes, \
                    comments \
                FROM access_information, created_by, channels, videos \
                WHERE ( \
                    access_information.video_id = created_by.video_id \
                    AND created_by.channel_id = channels.id \
                    AND access_information.video_id = videos.id \
                );\
            "
            )
            result = cursor.fetchall()

            ansi = getANSI("bold", "bright_blue")
            print()
            print(f"{ansi}filtering data...{resetANSI()}")
            rawData: np.ndarray = np.array(
                list(
                    filter(
                        lambda data: validThumbnail(self.imageDir, data[0])
                        and np.all(data),
                        tqdm(result),
                    )
                )
            )

        self.thumbnailIDs = rawData[:, 0:1]

        print(rawData[:, 3:4].dtype)
        print(rawData[:, -3:].dtype)
        self.data = np.concat(
            (
                self.secondsDifference(rawData[:, 2:3] - rawData[:, 1:2]).astype(
                    np.float32
                ),
                rawData[:, 3:4].astype(np.float32),
                rawData[:, -3:].astype(np.float32),
            ),
            1,
        )

        self.data = (self.data - np.mean(self.data, 0)) / np.std(self.data, 0)

        print()
        print(self.data.shape)
        print(self.data[:5])

        # load the thumbnails into a dictionary

        dbConnection.close()

    # endregion

    # region __len__
    def __len__(self):
        return len(self.data)

    # endregion

    # region __getitem__
    def __getitem__(self, idx):
        selectedImages = self.loadImages(self.thumbnailIDs[idx, 0])

        return torch.tensor(selectedImages), torch.tensor(self.data[idx, -3:])

    # endregion

    # region LoadImages
    def loadImages(self, vids):
        if type(vids) is str:
            return self.loadImage(vids)

        images = np.empty((len(vids), 3, self.imageSize[1], self.imageSize[0]))

        # find thumbnails
        threads = []
        for i in range(len(vids)):
            thread = Thread(target=self.loadImage, args=(vids[i], images, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return images

    # endregion

    # region LoadImage
    def loadImage(self, vid, output=None, i=0):
        imagePath = f"{self.imageDir}/{vid}.jpg"
        image = PIL.Image.open(imagePath)
        result = mplToNp(np.array(image.resize(self.imageSize), dtype=np.float32) / 255)

        if output is not None:
            output[i] = result

        return result

    # endregion


# endregion
