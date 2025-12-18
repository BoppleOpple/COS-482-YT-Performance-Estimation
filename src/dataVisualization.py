import os
import argparse
import datetime
from dotenv import load_dotenv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import psycopg2

# region arguments
parser = argparse.ArgumentParser(
    prog="python3 run.py",
    description="Tunes and Trains a neural network on YouTube data",
    epilog="Copyright Liam Hillery, 2025",
)

parser.add_argument_group("File I/O")
parser.add_argument("-i", "--imageDir", default="./output/thumbnails", type=Path)
parser.add_argument("-o", "--outDir", default="./output", type=Path)
parser.add_argument("-s", "--sessionName", default=None, type=str)

parser.add_argument_group("Training options")
parser.add_argument("-e", "--epochs", default=5, type=int)
parser.add_argument("--validation_epochs", default=None, type=int)
# endregion


# region subs_views_scatter
def subs_views_scatter(args, cursor: psycopg2.extensions.cursor):
    cursor.execute(
        "\
        SELECT \
            subscribers, \
            access_information.views \
        FROM access_information, created_by, channels, videos \
        WHERE ( \
            access_information.video_id = created_by.video_id \
            AND created_by.channel_id = channels.id \
            AND access_information.video_id = videos.id \
                AND subscribers IS NOT NULL \
        );\
    ",
    )
    data: np.ndarray = np.array(cursor.fetchall())

    x = data[:, 0]
    y = data[:, 1]

    fig = plt.figure(figsize=(12, 16), dpi=300)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([pow(10, n) for n in range(0, int(np.ceil(np.log10(np.max(x)))))])

    ax.set_title("Views vs. Subscriber Count")
    ax.set_xlabel("Subscribers")
    ax.set_ylabel("Views")

    ax.grid(True, linestyle="dashdot")

    # plt.plot(range(len(v_losses)), v_losses, label="validation", color="#fa96c8")
    plt.scatter(x, y, s=5, color="#6496fa60")

    fig.savefig(args.outDir / "subs_vs_views.png")


# endregion


# region uptime_views_scatter
def uptime_views_scatter(args, cursor: psycopg2.extensions.cursor):
    cursor.execute(
        "\
        SELECT DISTINCT ON (videos.id) \
            posted_time, \
            query_time, \
            access_information.views \
        FROM access_information, created_by, channels, videos \
        WHERE ( \
            access_information.video_id = created_by.video_id \
            AND created_by.channel_id = channels.id \
            AND access_information.video_id = videos.id \
        ) \
        ORDER BY videos.id, access_information.views DESC;\
    ",
    )
    data: np.ndarray = np.array(cursor.fetchall())

    @np.vectorize()
    def getDurationSeconds(d: datetime.timedelta):
        return d.total_seconds()

    x = getDurationSeconds(data[:, 1] - data[:, 0]) / (60 * 60 * 24)
    y = data[:, 2]

    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_xticks(range(0, int(max(x))))
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    ax.set_title("Views vs. Uptime")
    ax.set_xlabel("Time since upload (days)")
    ax.set_ylabel("Views")

    # plt.plot(range(len(v_losses)), v_losses, label="validation", color="#fa96c8")
    plt.scatter(x, y, s=5, color="#6496fa60")

    fig.savefig(args.outDir / "uptime_vs_views.png")


# endregion


# region time_views_scatter
def time_views_scatter(args, cursor: psycopg2.extensions.cursor):
    cursor.execute(
        "\
        SELECT DISTINCT ON (videos.id) \
            posted_time, \
            access_information.views \
        FROM access_information, created_by, channels, videos \
        WHERE ( \
            access_information.video_id = created_by.video_id \
            AND created_by.channel_id = channels.id \
            AND access_information.video_id = videos.id \
        ) \
        ORDER BY videos.id, access_information.views DESC;\
    ",
    )
    data: np.ndarray = np.array(cursor.fetchall())

    @np.vectorize()
    def getSecondsSinceMidnight(d: datetime.datetime):
        return (d.hour * 60 + d.minute) * 60 + d.second

    x = getSecondsSinceMidnight(data[:, 0]) / (60 * 60)
    y = data[:, 1]

    fig = plt.figure(figsize=(12, 16), dpi=300)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_xticks(range(0, 24))
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # plt.plot(range(len(v_losses)), v_losses, label="validation", color="#fa96c8")
    plt.scatter(x, y, s=5, color="#6496fa60")

    fig.savefig(args.outDir / "time_vs_views.png")


# endregion


# region subs_hist
def subs_hist(args, cursor: psycopg2.extensions.cursor, percentCaptured: float = 0.1):
    cursor.execute(
        "\
        SELECT \
            subscribers \
        FROM channels \
        WHERE ( \
            subscribers IS NOT NULL \
        );\
    ",
    )
    data: np.ndarray = np.array(cursor.fetchall())

    fig = plt.figure(dpi=300)

    data = data[data < np.quantile(data, percentCaptured)]
    plt.hist(data, bins=100)

    fig.savefig(args.outDir / f"subs_hist_lower_{percentCaptured}.png")


# endregion


# region views_hist
def views_hist(args, cursor: psycopg2.extensions.cursor, percentCaptured: float = 1.0):
    cursor.execute(
        "\
        SELECT \
            views \
        FROM access_information \
        WHERE ( \
            views IS NOT NULL \
        );\
    ",
    )
    data: np.ndarray = np.array(cursor.fetchall())

    fig = plt.figure(dpi=300)

    data = data[data < np.quantile(data, percentCaptured)]
    plt.hist(data, bins=100)

    fig.savefig(args.outDir / f"views_hist_lower_{percentCaptured}.png")


# endregion


# region time_hist
def time_hist(args, cursor: psycopg2.extensions.cursor):
    cursor.execute(
        "\
        SELECT DISTINCT ON (videos.id) \
            posted_time \
        FROM access_information, created_by, channels, videos \
        WHERE ( \
            access_information.video_id = created_by.video_id \
            AND created_by.channel_id = channels.id \
            AND access_information.video_id = videos.id \
        ) \
        ORDER BY videos.id, access_information.views DESC;\
    ",
    )
    data: np.ndarray = np.array(cursor.fetchall())

    @np.vectorize()
    def getSecondsSinceMidnight(d: datetime.datetime):
        return (d.hour * 60 + d.minute) * 60 + d.second

    x = getSecondsSinceMidnight(data) / (60 * 60)

    fig = plt.figure(dpi=300)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_xticks(range(0, 24))

    plt.hist(x, bins=120)

    fig.savefig(args.outDir / "time_hist.png")


# endregion


# region Main Execution
def main(argv=None):
    load_dotenv()
    args = parser.parse_args(argv)

    dbConnection = psycopg2.connect(
        host=os.environ["SQL_HOST"],
        port=os.environ["SQL_PORT"],
        dbname=os.environ["SQL_DBNAME"],
        user=os.environ["SQL_USER"],
        password=os.environ["SQL_PASSWORD"],
    )

    with dbConnection.cursor() as cursor:
        subs_views_scatter(args, cursor)
        time_views_scatter(args, cursor)
        uptime_views_scatter(args, cursor)

        subs_hist(args, cursor)
        views_hist(args, cursor, percentCaptured=0.1)
        time_hist(args, cursor)

    dbConnection.commit()
    dbConnection.close()


# endregion


# region Entry Point
if __name__ == "__main__":
    main()
# endregion
