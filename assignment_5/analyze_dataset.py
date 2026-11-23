from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATASET_DIR = Path("./dataset")

# Find all category directories inside the dataset folder.
categories = sorted(
    # Iterate through every file and directory inside DATASET_DIR.
    directory for directory in DATASET_DIR.iterdir()
    if directory.is_dir() and directory.name != "cifar-10-batches-py"
)

print(f"Number of categories: {len(categories)}")

# Go through each category directory to count its images.
for category in categories:
    images = list(category.glob("*.png"))# Find all PNG files in the current category and store them in a list.
    print(f"{category.name}: {len(images)} images")


# Create a figure with 2 rows and 5 columns to display 10 sample images.
# figsize=(12, 5) sets the figure dimensions in inches.
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

# Pair each subplot (ax) with a category directory.
# axes.flat converts the 2D subplot grid into a sequence of 10 axes.
# zip() pairs each axis with one category.
for ax, category in zip(axes.flat, categories):

    # Find all PNG images in the category, sort their filenames,
    # and select the first image using index [0].
    image_path = sorted(category.glob("*.png"))[0]

    # Open the selected PNG file as a PIL image.
    image = Image.open(image_path)

    # Display the image in the current subplot.
    ax.imshow(image)

    # Set the subplot title to the category name.
    ax.set_title(category.name)

    # Hide the x-axis, y-axis, and tick labels for a cleaner display.
    ax.axis("off")

# Automatically adjust spacing to prevent images and titles from overlapping.
plt.tight_layout()

# Save the complete 2×5 image grid as a PNG file in the current directory.
plt.savefig("dataset_samples.png")