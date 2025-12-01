import os
from threading import Thread, Semaphore
from tqdm import tqdm
from dotenv import load_dotenv
import requests

import psycopg2

outdir = "./output/thumbnails"
maxRequests = Semaphore(512)


# region downloadThumbnail
def downloadThumbnail(vid: str, url: str):
    extension = url.split(".")[-1]
    outfile = f"{outdir}/{vid}.{extension}"

    if not os.path.exists(outfile):
        maxRequests.acquire()
        thumbnailRequest = requests.get(url)

        with open(outfile, "wb") as f:
            f.write(thumbnailRequest.content)

        thumbnailRequest.close()
        maxRequests.release()


# endregion


# region Main Execution
def main():
    load_dotenv()

    dbConnection = psycopg2.connect(
        host=os.environ["SQL_HOST"],
        port=os.environ["SQL_PORT"],
        dbname=os.environ["SQL_DBNAME"],
        user=os.environ["SQL_USER"],
        password=os.environ["SQL_PASSWORD"],
    )

    result = None
    with dbConnection.cursor() as cursor:

        cursor.execute("SELECT id, thumbnail FROM videos;")
        result = cursor.fetchall()

    dbConnection.commit()
    dbConnection.close()

    os.makedirs(outdir, exist_ok=True)
    threads = []

    print("creating threads...")
    for thumbnailData in tqdm(result):
        thread = Thread(target=downloadThumbnail, args=thumbnailData)
        threads.append(thread)
        thread.start()

    print("joining threads...")
    for thread in tqdm(threads):
        thread.join()


# endregion


# region Entry Point
if __name__ == "__main__":
    main()
# endregion
