import os
import json
import string

lyrics_dir = "lyrics"
json_files = [file for file in os.listdir(lyrics_dir) if file.endswith(".json")]
merged_file = open(r"merged_lyrics_metalcore.txt", "a")
prefix="Lyrics_"
for file in json_files:
    print("Merging", file[file.startswith(prefix) and len(prefix):len(file) - 5], "lyrics into", merged_file.name)
    artist_lyrics = ""
    with open(os.path.join(lyrics_dir, file)) as json_file:
        text = json.load(json_file)
        for song in text["songs"]:
            # Append all artist lyrics into one string
            artist_lyrics += song["lyrics"].lower() + "\n"
    # Append artist lyrics to the merged file holding all of the artists' lyrics
    merged_file.write(artist_lyrics)

merged_file.close()
