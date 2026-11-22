"""
Dataset Preparation — CIFAR-10

Downloads CIFAR-10 and selects 210 images (21 per category).
Images are organized into category-specific directories.

Category labels are used only for balanced dataset selection,
not for feature extraction or clustering.

Source: https://www.cs.toronto.edu/~kriz/cifar.html
Reference: Krizhevsky, A. (2009).
           Learning Multiple Layers of Features from Tiny Images.
"""

from pathlib import Path
from torchvision.datasets import CIFAR10

# Configuration
DATASET_DIR = Path("./dataset")
IMAGES_PER_CATEGORY = 21


# Step 1: Download and load the CIFAR-10 training dataset.
dataset = CIFAR10(
    root=str(DATASET_DIR),
    train=True,
    download=True
)

print(f"Total available training images: {len(dataset)}")

# Step 2: Select 21 images from each category.
for label in range(len(dataset.classes)):
    count = 0
    category = dataset.classes[label]

    # Create a separate directory for each category.
    category_dir = DATASET_DIR / category
    category_dir.mkdir(exist_ok=True)

    for image, image_label in dataset:
        if image_label == label:
            filename = f"{category}_{count}.png"
            image.save(category_dir / filename)

            count += 1

            if count == IMAGES_PER_CATEGORY:
                break

    if count != IMAGES_PER_CATEGORY:
        raise RuntimeError(
            f"Expected {IMAGES_PER_CATEGORY} images for "
            f"{category}, but found {count}."
        )

    print(f"{category}: {count} images")

# Step 3: Display the dataset summary.
total_images = len(dataset.classes) * IMAGES_PER_CATEGORY

print("\nDataset Selection Summary")
print("-" * 35)
print(f"Total selected images: {total_images}")
print(f"Number of categories: {len(dataset.classes)}")
print(f"Images per category: {IMAGES_PER_CATEGORY}")
print("Image resolution: 32 x 32")
print(f"Output directory: {DATASET_DIR}")