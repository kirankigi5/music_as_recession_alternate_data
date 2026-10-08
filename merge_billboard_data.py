import csv
import json
import urllib.request

URL = "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/all.json"
OUT = "billboard_hot100_all.csv"
COLUMNS = ["date", "song", "artist", "this_week", "last_week", "peak_position", "weeks_on_chart"]

with urllib.request.urlopen(URL) as response:
    charts = json.load(response)

with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(COLUMNS)
    for chart in charts:
        for song in chart["data"]:
            writer.writerow([chart["date"]] + [song.get(col) for col in COLUMNS[1:]])