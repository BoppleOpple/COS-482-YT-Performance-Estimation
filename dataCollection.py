import os
import re
import time
import datetime
from dotenv import load_dotenv

import googleapiclient.discovery
import googleapiclient.errors

import psycopg2

descriptionLength = 5000
titleLength = 100
tagLength = 300
channelNameLength = 100
categoryLength = 200


def waitUntilTime(timeStr):
    targetHours, targetMinutes, targetSeconds = map(
        lambda s: float(s), timeStr.split(":")[:3]
    )

    print(targetHours, targetMinutes, targetSeconds)

    _, _, _, hours, minutes, seconds, *_ = time.localtime()

    secondsRemaining = (
        ((targetHours - hours) * 60 + targetMinutes - minutes) * 60
        + targetSeconds
        - seconds
    ) % (24 * 60 * 60)

    durationHours = int(secondsRemaining / (60 * 60) % 24)
    durationMinutes = int(secondsRemaining / 60 % 60)
    durationSeconds = int(secondsRemaining % 60)
    print(
        f"Time remaining: {durationHours:02}:{durationMinutes:02}:{durationSeconds:02}"
    )

    time.sleep(secondsRemaining)


def parseISOTimestamp(isoString):
    match = re.match(r"(\d+)-(\d+)-(\d+)T(\d+):(\d+):([0123456789.]+)Z", isoString)

    return match.groups()


def parseDuration(durationString):
    hourMatch = re.search(r"\d+(?=H)", durationString)
    minuteMatch = re.search(r"\d+(?=M)", durationString)
    secondMatch = re.search(r"\d+(?=S)", durationString)

    hours = 0
    minutes = 0
    seconds = 0

    if hourMatch:
        hours = int(hourMatch.group(0))

    if minuteMatch:
        minutes = int(minuteMatch.group(0))

    if secondMatch:
        seconds = int(secondMatch.group(0))

    return 3600 * hours + 60 * minutes + seconds


def handleChannelResponse(rid, response, exception, channelIDs):

    if exception:
        print(exception)

    if response and response["pageInfo"]["totalResults"] != 0:
        return {
            "id": response["items"][0]["id"],
            "name": response["items"][0]["snippet"]["title"],
            "views": response["items"][0]["statistics"]["viewCount"],
            "subscribers": (
                None
                if response["items"][0]["statistics"]["hiddenSubscriberCount"]
                else response["items"][0]["statistics"]["subscriberCount"]
            ),
            "videos": response["items"][0]["statistics"]["videoCount"],
        }
    else:
        return {
            "id": channelIDs[rid][0],
            "name": channelIDs[rid][1],
            "views": None,
            "subscribers": None,
            "videos": None,
        }


def handleResponses(jsonResponses, channels, category):
    queryTime = datetime.datetime.now(datetime.UTC).isoformat()
    # queryTime = "2025-10-29 01:12:28"

    dbConnection = psycopg2.connect(
        host=os.environ["SQL_HOST"],
        port=os.environ["SQL_PORT"],
        dbname=os.environ["SQL_DBNAME"],
        user=os.environ["SQL_USER"],
        password=os.environ["SQL_PASSWORD"],
    )

    with dbConnection.cursor() as cursor:

        cursor.execute(
            "INSERT INTO categories (id, name) \
            VALUES (%s, %s) \
            ON CONFLICT DO NOTHING;",
            (
                category["id"],
                category["snippet"]["title"][:categoryLength],
            ),
        )

        cursor.execute(
            "INSERT INTO queries (category_id, time) \
            VALUES (%s, %s) \
            ON CONFLICT DO NOTHING;",
            (
                category["id"],
                queryTime,
            ),
        )

        for channel in channels:
            cursor.execute(
                "INSERT INTO channels (id, name, views, subscribers, videos) \
                VALUES (%s, %s, %s, %s, %s) \
                ON CONFLICT (id) DO UPDATE SET ( \
                    name, \
                    views, \
                    subscribers, \
                    videos \
                ) = ( \
                    EXCLUDED.name, \
                    EXCLUDED.views, \
                    EXCLUDED.subscribers, \
                    EXCLUDED.videos \
                );",
                (
                    channel["id"],
                    channel["name"][:channelNameLength],
                    channel["views"],
                    channel["subscribers"],
                    channel["videos"],
                ),
            )

        for response in jsonResponses:
            for video in response["items"]:
                thumbnailResolution = list(video["snippet"]["thumbnails"].keys())[-1]

                cursor.execute(
                    "INSERT INTO videos ( \
                        id, \
                        title, \
                        description, \
                        thumbnail, \
                        duration, \
                        posted_time \
                    ) VALUES (%s, %s, %s, %s, %s, %s) \
                    ON CONFLICT DO NOTHING;",
                    (
                        video["id"],
                        video["snippet"]["title"][:titleLength],
                        video["snippet"]["description"][:descriptionLength],
                        video["snippet"]["thumbnails"][thumbnailResolution]["url"],
                        parseDuration(video["contentDetails"]["duration"]),
                        video["snippet"]["publishedAt"],
                    ),
                )

                cursor.execute(
                    "INSERT INTO access_information ( \
                    video_id, \
                    query_category, \
                    query_time, \
                    views, \
                    likes, \
                    comments \
                    ) VALUES (%s, %s, %s, %s, %s, %s) \
                    ON CONFLICT DO NOTHING;",
                    (
                        video["id"],
                        category["id"],
                        queryTime,
                        (
                            video["statistics"]["viewCount"]
                            if ("viewCount" in video["statistics"].keys())
                            else None
                        ),
                        (
                            video["statistics"]["likeCount"]
                            if ("likeCount" in video["statistics"].keys())
                            else None
                        ),
                        (
                            video["statistics"]["commentCount"]
                            if ("commentCount" in video["statistics"].keys())
                            else None
                        ),
                    ),
                )

                cursor.execute(
                    "INSERT INTO created_by (video_id, channel_id) \
                    VALUES (%s, %s) \
                    ON CONFLICT DO NOTHING;",
                    (
                        video["id"],
                        video["snippet"]["channelId"],
                    ),
                )

                try:
                    for tag in video["snippet"]["tags"]:
                        cursor.execute(
                            "INSERT INTO tags (tag) \
                            VALUES (%s) \
                            ON CONFLICT DO NOTHING;",
                            (tag[:tagLength],),
                        )

                        cursor.execute(
                            "INSERT INTO has_tag (video_id, tag) \
                            VALUES (%s, %s) \
                            ON CONFLICT DO NOTHING;",
                            (
                                video["id"],
                                tag[:tagLength],
                            ),
                        )
                except KeyError:
                    pass

    dbConnection.commit()
    dbConnection.close()


def main():
    load_dotenv()

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=os.environ["GOOGLE_API_KEY"]
    )

    categoryRequest = youtube.videoCategories().list(
        part="snippet", regionCode="US", hl="en"
    )

    categories = [
        {
            "kind": "youtube#videoCategory",
            "id": "0",
            "snippet": {"title": "Uncategorized"},
        }
    ]

    categoryResponse = categoryRequest.execute()
    categories.extend(categoryResponse["items"])

    categoryNumber = 1
    for category in categories:

        videoRequest = youtube.videos().list(
            part="snippet,contentDetails,statistics",  # maybe add topic details
            chart="mostPopular",
            regionCode="US",
            hl="en",
            maxResults=50,
            videoCategoryId=category["id"],
        )

        videoResponses = []
        page = 1

        while videoRequest is not None:
            print(f"fetching video page {page}")
            try:
                videoResponse = videoRequest.execute()

                videoResponses.append(videoResponse)

                videoRequest = youtube.videos().list_next(
                    previous_request=videoRequest, previous_response=videoResponse
                )
                page += 1
            except googleapiclient.errors.HttpError:
                print("http error???")
                break

        channels = []
        channelIDs = []

        request = youtube.new_batch_http_request(
            lambda rid, response, exception: channels.append(
                handleChannelResponse(int(rid), response, exception, channelIDs)
            )
        )

        i = 0
        for response in videoResponses:
            for video in response["items"]:
                channelIDs.append(
                    (video["snippet"]["channelId"], video["snippet"]["channelTitle"])
                )

                request.add(
                    youtube.channels().list(
                        part="snippet,statistics",
                        id=video["snippet"]["channelId"],
                        hl="en",
                    ),
                    request_id=str(i),
                )
                i += 1

        request.execute()

        categoryName = category["snippet"]["title"]
        print(
            f"Collected data from YT in category {categoryName} ({categoryNumber}/{len(categories)})"  # noqa: E501
        )

        categoryNumber += 1

        handleResponses(videoResponses, channels, category)


if __name__ == "__main__":
    while True:
        waitUntilTime("15:00:00")
        main()
