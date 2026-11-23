from pathlib import Path
import numpy as np
from collections import Counter

images = sorted(Path("dataset").glob("*/*.png"))
labels = np.load("cluster_labels.npy")

if len(images) != len(labels):
    raise ValueError("Image and label counts do not match.")

cluster_id = Counter(labels).most_common(1)[0][0]
indices = np.where(labels == cluster_id)[0]


categories = []

for i in indices:
    categories.append(images[i].parent.name)

dominant = Counter(categories).most_common(1)[0][0]

music_mapping = {  "automobile": "energetic","truck": "energetic","airplane": "cinematic","ship": "cinematic","cat": "calm","dog": "calm","bird": "calm","deer": "calm","horse": "calm", "frog": "calm"
}

print("Selected cluster:", cluster_id)
print("Dominant category:", dominant)
music_style = music_mapping.get(dominant, "calm")
print("Selected music style:", music_style)


music_tracks = {
    "energetic": "Danza Kuduro",
    "cinematic": "Cinematic Instrumental",
    "calm": "Calm Ambient"
}
selected_song = music_tracks[music_style]
print("Selected song:", selected_song)