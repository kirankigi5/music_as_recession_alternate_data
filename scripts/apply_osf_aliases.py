import csv
from collections import Counter
from pathlib import Path

def norm(value):
    return (value or "").strip().casefold()

alias_path = Path("config/osf_artist_aliases.csv")
inventory_path = Path("data/metadata/song_lyrics_inventory.csv")

with alias_path.open(encoding="utf-8", newline="") as f:
    aliases = {
        (norm(r["song"]), norm(r["osf_artist"])):
        (norm(r["song"]), norm(r["billboard_artist"]))
        for r in csv.DictReader(f)
    }

with open("billboard_hot100_all.csv",
          encoding="utf-8-sig", newline="") as f:
    charts = {
        (r["date"], r["this_week"]):
        (norm(r["song"]), norm(r["artist"]))
        for r in csv.DictReader(f)
    }

evidence = Counter()

with open("data/raw/lyrics/osf_lyrics.csv",
          encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        source = (norm(r["Song Title"]), norm(r["Artist"]))
        target = aliases.get(source)

        if target is None:
            continue

        slot = (r["Date"], r["Rank"])
        if charts.get(slot) != target:
            continue

        lyrics = (r.get("lyrics_raw") or "").strip()
        if len(lyrics) >= 20 and norm(lyrics) not in {
            "nan", "none", "null"
        }:
            evidence[target] += 1

with inventory_path.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    inventory = list(reader)

updated = 0

for r in inventory:
    key = (norm(r["song"]), norm(r["artist"]))

    if evidence[key] and r["lyrics_status"] == "not_collected":
        r["lyrics_status"] = "matched_osf_alias_provisional"
        r["lyrics_source"] = "OSF:2k7ut"
        r["match_notes"] = (
            "Chart date, rank, title aligned; "
            "artist alias and recording need review"
        )
        updated += 1

temporary = inventory_path.with_suffix(".tmp")

with temporary.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(inventory)

temporary.replace(inventory_path)

total_matched = sum(
    r["lyrics_status"].startswith("matched_osf")
    for r in inventory
)

print("Alias mappings:", len(aliases))
print("Aliases supported by chart records:",
      sum(evidence[a] > 0 for a in aliases.values()))
print("New provisional matches:", updated)
print("Total candidate lyric matches:", total_matched)
print("Remaining inventory songs:",
      len(inventory) - total_matched)
print("Inventory updated:", inventory_path)
