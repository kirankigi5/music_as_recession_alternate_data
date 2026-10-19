import csv
from collections import defaultdict
from pathlib import Path

billboard_path = Path("billboard_hot100_all.csv")
lyrics_path = Path("data/raw/lyrics/osf_lyrics.csv")
inventory_path = Path("data/metadata/song_lyrics_inventory.csv")

def song_key(title, artist):
    return (title.strip().casefold(), artist.strip().casefold())

# Build exact chart-date/rank lookup.
charts = {}
year_entries = defaultdict(list)

with billboard_path.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        key = song_key(row["song"], row["artist"])
        slot = (row["date"], row["this_week"])
        charts[slot] = key
        year_entries[row["date"][:4]].append(
            (key, 101 - int(row["this_week"]))
        )

print("Billboard entries indexed:", sum(map(len, year_entries.values())), flush=True)

matched_songs = set()
total_rows = 0
exact_rows = 0
rows_with_lyrics = 0

with lyrics_path.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        total_rows += 1
        slot = (row["Date"], row["Rank"])
        key = song_key(row["Song Title"], row["Artist"])

        if charts.get(slot) == key:
            exact_rows += 1
            lyrics = (row.get("lyrics_raw") or "").strip()

            if len(lyrics) >= 20 and lyrics.casefold() not in {
                "nan", "none", "null"
            }:
                rows_with_lyrics += 1
                matched_songs.add(key)

        if total_rows % 25000 == 0:
            print(f"Processed {total_rows:,} OSF records...", flush=True)

# Update metadata only. Never write lyrics into inventory.
with inventory_path.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    inventory = list(reader)

matched_inventory = 0

for row in inventory:
    key = song_key(row["song"], row["artist"])

    if key in matched_songs:
        matched_inventory += 1
        row["lyrics_status"] = "matched_osf_unverified"
        row["lyrics_source"] = "OSF:2k7ut"

temporary = inventory_path.with_suffix(".tmp")

with temporary.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(inventory)

temporary.replace(inventory_path)

print("\n=== OSF MATCHING RESULTS ===")
print("OSF records scanned:", f"{total_rows:,}")
print("Exact chart-record matches:", f"{exact_rows:,}")
print("Exact matches containing lyrics:", f"{rows_with_lyrics:,}")
print("Unique songs with candidate lyrics:", f"{matched_inventory:,}")
print("Songs still missing:", f"{len(inventory)-matched_inventory:,}")
print("Total inventory songs:", f"{len(inventory):,}")
print("Song coverage:", f"{matched_inventory/len(inventory):.1%}")

print("\n=== YEARLY CHART COVERAGE ===")
print("Year | Entries covered | Popularity-weight coverage")

for year in ["1958", "1973", "1980", "1990",
             "2000", "2010", "2020", "2023", "2024", "2026"]:
    entries = year_entries.get(year, [])
    covered = sum(key in matched_songs for key, _ in entries)
    total_weight = sum(weight for _, weight in entries)
    covered_weight = sum(
        weight for key, weight in entries if key in matched_songs
    )
    pct = covered_weight / total_weight if total_weight else 0
    print(f"{year} | {covered:,}/{len(entries):,} | {pct:.1%}")

print("\nUpdated:", inventory_path)
