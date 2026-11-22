
"""
Visualize all K-Means clusters to evaluate image similarities.
"""

from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


image_paths = sorted(Path("./dataset").glob("*/*.png"))  # Find images in their original feature extraction order.
cluster_labels = np.load("cluster_labels.npy")            # Load the cluster IDs assigned by K-Means.

print(f"Total images: {len(image_paths)}")
print(f"Total cluster assignments: {len(cluster_labels)}")

assert len(image_paths) == len(cluster_labels), "Image and label counts do not match."  # Verify one label per image.


for cluster_id in sorted(np.unique(cluster_labels)):  # Process each cluster automatically.

    cluster_indices = np.where(cluster_labels == cluster_id)[0]  # Find image positions belonging to this cluster.

    print(f"Cluster {cluster_id} contains {len(cluster_indices)} images")

    columns = 5                                        # Display a maximum of five images per row.
    rows = math.ceil(len(cluster_indices) / columns)    # Calculate the required number of rows.

    fig, axes = plt.subplots(rows, columns, figsize=(15, rows * 3))  # Create the image grid.

    for ax, index in zip(axes.flat, cluster_indices):  # Match each image to a grid cell.
        image = Image.open(image_paths[index])          # Open the original PNG image.
        ax.imshow(image)                                # Display the image.
        ax.set_title(image_paths[index].parent.name)    # Show its original CIFAR-10 category.
        ax.axis("off")                                  # Hide axis labels and ticks.

    for ax in list(axes.flat)[len(cluster_indices):]:   # Find unused grid cells.
        ax.axis("off")                                  # Hide those empty cells.

    plt.tight_layout()                                  # Prevent images and titles from overlapping.
    plt.savefig(f"cluster_{cluster_id}.png")             # Save the current cluster visualization.
    plt.close(fig)                                      # Release resources before the next cluster.

print("All cluster visualizations saved successfully.")
