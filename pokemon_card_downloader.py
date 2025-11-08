import requests
import os
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

# base folder to download all the cards in
BASE_FOLDER = "pokemon_cards_2011_2023"
os.makedirs(BASE_FOLDER, exist_ok=True)

# global variables just bc its so annoying constantly changing them when they get errors downloading
MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds

#obtaining all the sets
sets_url = "https://api.pokemontcg.io/v2/sets"
response = requests.get(sets_url)
all_sets = response.json().get("data", [])

# filtering sets by release date (OOps I accidentally downloaded 2017-2023 at first. Those are ALL cards with the IDs on the left side.)
filtered_sets = []
for s in all_sets:
    release_date = s.get("releaseDate") #syntax thankfully retrieved from API documentation https://docs.pokemontcg.io/api-reference/cards/get-card 
    if release_date:
        try:
            year = int(release_date.split("/")[0])
            if 2011 <= year <= 2016:
                filtered_sets.append((s["id"], release_date)) 
        except ValueError:
            continue

# sorting by release date-- the earliest will be downloaded first just to maintain easy organization 
filtered_sets.sort(key=lambda x: x[1])
sorted_set_ids = [s[0] for s in filtered_sets]

print(f"Found {len(sorted_set_ids)} sets from 2011-2023")

#downloading the card images
base_url = "https://api.pokemontcg.io/v2/cards"

for set_id in sorted_set_ids:
    set_folder = os.path.join(BASE_FOLDER, set_id)

    # skipping the set if folder already exists--I already downloaded the images of the set thus it has its own folder already
    if os.path.exists(set_folder):
        print(f"\nSkipping set {set_id} (folder exists)")
        continue

    print(f"\nDownloading cards from set: {set_id}")
    os.makedirs(set_folder, exist_ok=True)

    page = 1
    while True:
        url = f"{base_url}?q=set.id:{set_id}&page={page}&pageSize=100" #separating the sets by 100 into pages--it takes way too long when I tried opening it with 250 cards per page at a time

        # retrying when it fails because it kept constantly going into error and skipping folders
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                break
            except Exception as e:
                print(f"Attempt {attempt} failed for set {set_id} page {page}: {e}")
                if attempt == MAX_RETRIES:
                    print(f"Skipping set {set_id} page {page} after {MAX_RETRIES} attempts")
                    response = None
                else:
                    time.sleep(RETRY_DELAY)

        if response is None:
            break  

        cards = response.json().get("data", [])
        if not cards:
            break

        for card in cards:
            # fixing the filename
            name_clean = card["name"].replace("/", "_")
            filename = os.path.join(set_folder, f"{card['id']} - {name_clean}.png")

            # skipping duplicates
            if os.path.exists(filename):
                print(f"Skipped (already downloaded): {name_clean}")
                continue

            # retrying loop for when it goes into error when downloading the image
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    img_data = requests.get(card["images"]["large"], timeout=30).content
                    with open(filename, "wb") as f:
                        f.write(img_data)
                    print(f"Downloaded: {name_clean}")
                    break  # success
                except Exception as e:
                    print(f"Attempt {attempt} failed for image {name_clean}: {e}")
                    if attempt == MAX_RETRIES:
                        print(f"Failed to download {name_clean} after {MAX_RETRIES} attempts")
                    else:
                        time.sleep(RETRY_DELAY)

        page += 1
        time.sleep(0.5)

print("\nFinished downloading all cards from 2017-2023!")
