import os
import tqdm
from dotenv import load_dotenv
import requests

import psycopg2

outdir = "./output/thumbnails"

def main():
	load_dotenv()

	dbConnection = psycopg2.connect(
		host=os.environ["SQL_HOST"],
		port=os.environ["SQL_PORT"],
		dbname=os.environ["SQL_DBNAME"],
		user=os.environ["SQL_USER"],
		password=os.environ["SQL_PASSWORD"]
	)

	result = None
	with dbConnection.cursor() as cursor:
		
		cursor.execute(
			"SELECT id, thumbnail FROM videos LIMIT 5;"
		)
		result = cursor.fetchall()

	dbConnection.commit()
	dbConnection.close()

	os.makedirs(outdir, exist_ok=True)

	for thumbnailData in tqdm(result):
		thumbnailRequest = requests.get(thumbnailData[1])

		extension = thumbnailData[1][thumbnailData[1].rindex("."):]

		with open(f"{outdir}/{thumbnailData[0]}.{extension}", "wb") as f:
			f.write(thumbnailRequest.content)

		thumbnailRequest.close()

if __name__ == "__main__":
	main()
