import torch
from torchvision import models, transforms # this gives us the model we use
from PIL import Image # helps to open image
from pathlib import Path
import numpy as np

# Load ResNet-18 with weights learned from the ImageNet dataset.
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Remove the final classification layer to extract image features.
model.fc = torch.nn.Identity()

# Set the model to evaluation mode because we are not training it.
model.eval()

# Apply the preprocessing required by the pretrained ResNet-18 model.
preprocess = weights.transforms()

# Specify the directory containing our 10 image categories.
DATASET_DIR = Path("./dataset")

# Find all PNG images inside the category directories.
image_paths = sorted(DATASET_DIR.glob("*/*.png"))

# Display the total number of images found.
print(f"Total images found: {len(image_paths)}")



# Store the extracted feature vector for each image.
features = []

# Disable gradient calculations because we are only extracting features.
with torch.no_grad():
    # Process each image in the dataset.
    for image_path in image_paths:
        # Open the image and ensure it has three RGB color channels.
        image = Image.open(image_path).convert("RGB")
        # Apply ResNet-18 preprocessing and add a batch dimension.
        input_tensor = preprocess(image).unsqueeze(0)
        # Pass the image through ResNet-18 to extract its 512 features.
        feature_vector = model(input_tensor)
        # Convert the tensor into a NumPy array and store its features.
        features.append(feature_vector.squeeze(0).numpy())

print(f"Feature vectors extracted: {len(features)}")
print(f"Features per image: {len(features[0])}")


# Convert the list of feature vectors into a NumPy matrix.
feature_matrix = np.array(features)

# Save the feature matrix for the clustering stage.
np.save("features.npy", feature_matrix)

# Confirm the dimensions of the saved matrix.
print(f"Feature matrix shape: {feature_matrix.shape}")