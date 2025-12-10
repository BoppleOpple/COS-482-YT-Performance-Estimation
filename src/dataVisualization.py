import os
import argparse
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
        );\
    ",
    )
    data: np.ndarray = np.array(cursor.fetchall())

    fig = plt.figure(figsize=(12, 16), dpi=300)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_xscale("log")
    ax.set_yscale("log")

    # plt.plot(range(len(v_losses)), v_losses, label="validation", color="#fa96c8")
    plt.scatter(data[:, 0], data[:, 1], s=5, color="#6496fa60")

    fig.savefig(args.outDir / "subs_vs_views.png")


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
        subs_hist(args, cursor)
        views_hist(args, cursor, percentCaptured=0.1)

    dbConnection.commit()
    dbConnection.close()


# endregion


# region Entry Point
if __name__ == "__main__":
    main()
# endregion
