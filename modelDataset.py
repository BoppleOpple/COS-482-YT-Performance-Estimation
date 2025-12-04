import os
import psycopg2
import datetime
from threading import Thread
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

import torch
import torchtext
from torch.utils.data import Dataset, default_collate

import matplotlib.pyplot as plt
import PIL.Image

import dataResourceDownload
from printHelpers import printANSI

brokenThumbnailImage: PIL.ImageFile.ImageFile = PIL.Image.open(
    "res/brokenThumbnail.jpg"
)


# region Helpers
def npToMpl(image: np.ndarray):
    return np.swapaxes(np.swapaxes(image, -2, -1), -3, -1)


def mplToNp(image: np.ndarray):
    return np.swapaxes(np.swapaxes(image, -3, -1), -2, -1)


def validThumbnail(imageDir: str, vid: str):
    imagePath = f"{imageDir}/{vid}.jpg"
    return not brokenThumbnailImage == PIL.Image.open(imagePath)


@np.vectorize()
def secondsDifference(duration: datetime.timedelta):
    return duration.total_seconds()


# endregion


# region YTDataset
class YTDataset(Dataset):
    images = dict()
    textField = torchtext.data.Field(
        tokenize="spacy",
        tokenizer_language="en_core_web_sm",
        batch_first=True,
        pad_first=True,
    )

    # region __init__
    def __init__(
        self, imageDir="./output/thumbnails", imageSize=(640, 360), vocabSize=10000
    ):
        load_dotenv()

        self.imageSize = imageSize
        self.imageDir = imageDir
        self.vocabSize = vocabSize

        dbConnection: psycopg2.extensions.connection = psycopg2.connect(
            host=os.environ["SQL_HOST"],
            port=os.environ["SQL_PORT"],
            dbname=os.environ["SQL_DBNAME"],
            user=os.environ["SQL_USER"],
            password=os.environ["SQL_PASSWORD"],
        )

        print()
        printANSI("updating thumbnails...", "bold", "bright_blue")
        dataResourceDownload.main()

        # Download data from the database
        with dbConnection.cursor() as cursor:
            cursor.execute(
                "\
                SELECT \
                    access_information.video_id, \
                    title, \
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

            print()
            printANSI("filtering data...", "bold", "bright_blue")
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

        printANSI("building vocab...", "bold", "bright_blue")

        self.preprocessedText = list(map(self.textField.preprocess, rawData[:, 1]))

        self.textField.build_vocab(
            self.preprocessedText, max_size=vocabSize - 5, vectors="glove.6B.100d"
        )

        # self.data[0]: time since post
        # self.data[1]: subscriber count
        # self.data[2]: view count
        # self.data[3]: like count
        # self.data[4]: comment count
        self.data = np.concat(
            (
                secondsDifference(rawData[:, 3:4] - rawData[:, 2:3]).astype(np.float32),
                rawData[:, 4:5].astype(np.float32),
                rawData[:, -3:].astype(np.float32),
            ),
            axis=1,
        )

        self.data = (self.data - np.mean(self.data, 0)) / np.std(self.data, 0)

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

        return (
            torch.tensor(selectedImages),
            self.preprocessedText[idx],
            torch.tensor(self.data[idx, :-3]),
            torch.tensor(self.data[idx, -3:]),
        )

    # endregion

    def collate_fn(self, batch):
        titles = [sample[1] for sample in batch]

        # TODO: move this to the actual right part of the program
        paddedTitles = self.textField.pad(titles)
        numericalizedTitles = self.textField.numericalize(paddedTitles)
        expandedTitles = torch.zeros(
            (*numericalizedTitles.shape, self.vocabSize), dtype=torch.float32
        )

        for idx, val in np.ndenumerate(numericalizedTitles):
            expandedTitles[idx[0], idx[1], val] = 1

        for i in range(len(batch)):
            batch[i] = [*batch[i][:1], expandedTitles[i], *batch[i][2:]]

        return default_collate(batch)

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


# region showGrid
def showGrid(size, images, text=None, filename=None):
    figure, axs = plt.subplots(nrows=size[0], ncols=size[1])
    figure.tight_layout(pad=0)
    figure.set_size_inches(size[1] * 3, size[0] * 3)

    if images.shape[-1] != 3:
        images = npToMpl(images)

    for i in range(len(axs)):
        for j in range(len(axs[i])):
            axs[i][j].imshow(images[i * len(axs[i]) + j])
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            if text:
                axs[i][j].set_title(text[i * len(axs[i]) + j])

    plt.show()

    if filename:
        plt.savefig(filename, dpi=500)


# endregion
