# Image Clustering and Video Album Generation

## 1. Project Overview

This project groups similar images using machine learning and uses the resulting clusters to create a slideshow-style video with matching background music.

The system uses a pretrained Convolutional Neural Network (CNN) to extract image features, K-Means clustering to group similar images, and a rule-based algorithm to select appropriate music based on the dominant image category.

## 2. Dataset Collection

The dataset consists of **210 images** selected from the publicly available CIFAR-10 dataset.

CIFAR-10 contains images belonging to 10 categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

For this project, 21 images were selected from each category, resulting in 210 images.

Each image has a resolution of 32 × 32 pixels and contains three color channels: Red, Green, and Blue (RGB).

**Dataset source:** https://www.cs.toronto.edu/~kriz/cifar.html

## 3. Image Preprocessing

Before extracting features, the images are prepared for the pretrained neural network.

### RGB Image Representation

Each image is represented using three color channels:

- **Red (R):** Red intensity
- **Green (G):** Green intensity
- **Blue (B):** Blue intensity

Each pixel contains three numerical values representing its color.

For example, an RGB pixel can be represented as:

`(255, 0, 0)` — Red

`(0, 255, 0)` — Green

`(0, 0, 255)` — Blue

Since each image is 32 × 32 pixels, its original shape is:

`32 × 32 × 3`

### Image Conversion

The images are converted into numerical tensors so they can be processed by the neural network.

The preprocessing prepares the RGB images for ResNet18 using the transformations required by the pretrained model, including resizing and normalization.

Instead of directly comparing individual RGB pixel values, the system uses the neural network to extract more meaningful visual features.

## 4. Feature Extraction Using ResNet18

Feature extraction converts an image into a numerical representation that describes its visual characteristics.

This project uses **ResNet18**, a pretrained Convolutional Neural Network available through PyTorch's `torchvision` library.

ResNet18 has already learned useful visual patterns from a large image dataset, allowing it to recognize features such as shapes, textures, edges, and object structures.

The final classification layer is replaced with `torch.nn.Identity()` so the model returns feature representations rather than class predictions.

Each image is converted into a vector containing **512 numerical features**.

For 210 images, the resulting feature matrix has the shape:

`(210, 512)`

These features are saved in:

`features.npy`

Using pretrained ResNet18 avoids training a new neural network from scratch and provides useful representations for comparing images.

## 5. Image Clustering Using K-Means

After feature extraction, K-Means clustering groups images based on similarities in their feature vectors.

The algorithm is configured with:

- Number of clusters: 10
- Random state: 42
- Number of initializations: 10

K-Means works by assigning each image to a cluster whose center is closest to its feature representation. The algorithm repeatedly updates the cluster centers until the assignments stabilize.

The resulting cluster labels are saved in:

`cluster_labels.npy`

Each label identifies the cluster assigned to the corresponding image.

## 6. Cluster Visualization and Analysis

To examine the clustering results, images belonging to each cluster are displayed together in a grid.

The program generates visualization files such as:

`cluster_0.png`, `cluster_1.png`, ..., `cluster_9.png`

These images help evaluate whether the clustering algorithm grouped visually similar objects together.

### Clustering Results

| Cluster | Number of Images |
|---|---:|
| 0 | 9 |
| 1 | 25 |
| 2 | 22 |
| 3 | 32 |
| 4 | 25 |
| 5 | 22 |
| 6 | 15 |
| 7 | 22 |
| 8 | 27 |
| 9 | 11 |
| **Total** | **210** |

Cluster 3 contains the largest number of images.

Its composition is:

| Category | Number of Images |
|---|---:|
| Automobile | 15 |
| Ship | 10 |
| Truck | 6 |
| Airplane | 1 |
| **Total** | **32** |

### Observations

Cluster 3 contains mostly transportation-related images, including automobiles, ships, trucks, and an airplane.

Although these objects belong to different CIFAR-10 categories, they share some visual characteristics, which may explain why K-Means grouped them together.

The results demonstrate that feature-based clustering can identify similarities between images without directly using their original category labels to form the clusters.

However, not every cluster necessarily corresponds to a single original category.

## 7. Automatic Cluster Selection

The system automatically identifies the largest cluster using Python's `Counter` class.

The selected cluster is determined by counting how many images belong to each cluster.

For this dataset, the selected cluster is:

`Cluster 3 — 32 images`

The system then retrieves the original dataset category of each image in the selected cluster.

Using `Counter` again, it determines the most common category.

The dominant category for Cluster 3 is:

`automobile`

## 8. Algorithmic Music Selection

The project uses a rule-based algorithm to select background music based on the visual content of the selected cluster.

The music selection process consists of three steps.

### Step 1: Identify the Dominant Category

The algorithm counts the original image categories within the selected cluster.

For Cluster 3, automobiles are the most common category.

### Step 2: Determine the Music Style

Each category is associated with a predefined music style.

| Image Category | Music Style |
|---|---|
| Automobile, Truck | Energetic |
| Airplane, Ship | Cinematic |
| Cat, Dog, Bird, Deer, Horse, Frog | Calm |

Since the dominant category is automobile, the selected style is **energetic**.

### Step 3: Select the Song

The music style is mapped to a specific song.

| Music Style | Selected Song |
|---|---|
| Energetic | Danza Kuduro |
| Cinematic | Cinematic Instrumental |
| Calm | Calm Ambient |

The program automatically produces:

```text
Selected cluster: 3
Dominant category: automobile
Selected music style: energetic
Selected song: Danza Kuduro
```

The music selection is algorithmic rather than random because the song is determined through predefined rules based on the dominant category of the selected cluster.

## 9. Video Creation

The selected cluster is used to create a slideshow-style video.

Cluster 3 was selected because it contains the largest number of images.

The slideshow is assembled using CapCut, with images from the selected cluster displayed sequentially.

The background song is chosen by the Python algorithm and then added to the video during editing.

For Cluster 3, the algorithm selects *Danza Kuduro* because the dominant category is automobile, which maps to energetic music.

The clustering and music selection are performed programmatically, while the slideshow assembly and audio integration are performed manually in CapCut.

## 10. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Store and process numerical feature vectors and cluster labels |
| PyTorch | Run the pretrained neural network |
| torchvision | Load ResNet18 and perform image transformations |
| scikit-learn | K-Means clustering |
| Matplotlib | Visualize images and clusters |
| pathlib | Access image files and directories |
| collections.Counter | Identify the largest cluster and dominant category |
| CapCut | Assemble the slideshow and add background music |

## 11. Project Outputs

The project produces the following outputs:

- `dataset/` — Original images organized by category
- `features.npy` — Extracted 512-dimensional image features
- `cluster_labels.npy` — Cluster assignments
- `cluster_0.png` through `cluster_9.png` — Cluster visualizations
- Music selection output — Selected cluster, dominant category, music style, and song
- Final slideshow video — Video created from the selected cluster with matching background music

## Installation and Execution

### Install Dependencies

```bash
pip install numpy torch torchvision scikit-learn matplotlib pillow
```

### Run the Scripts

Execute the following commands from the project directory in order.

**1. Download Dataset**
```bash
python download_dataset.py
```

**2. Analyze Dataset**
```bash
python analyze_dataset.py
```

**3. Extract Features**
```bash
python extract_features.py
```

**4. Perform K-Means Clustering**
```bash
python cluster_images.py
```

**5. Visualize Clusters**
```bash
python visualize_clusters.py
```

**6. Select Background Music**
```bash
python select_music.py
```

### Generated Outputs

| File | Description |
|---|---|
| `dataset/` | 210 CIFAR-10 images |
| `dataset_samples.png` | Dataset sample visualization |
| `features.npy` | Extracted 512-dimensional image features |
| `cluster_labels.npy` | K-Means cluster assignments |
| `cluster_0.png` – `cluster_9.png` | Cluster visualizations |

### Music Selection Output

```text
Selected cluster: 3
Dominant category: automobile
Selected music style: energetic
Selected song: Danza Kuduro
```

### Video Creation

Create a slideshow in CapCut using images from the selected cluster. Add the algorithm-selected song and export the video in MP4 format.

# 12. Conclusion

This project demonstrates how machine learning can be used to organize images into meaningful groups and select suitable background music.

A dataset of 210 RGB images was processed using pretrained ResNet18 to extract 512-dimensional feature vectors. K-Means clustering was then applied to divide the images into 10 clusters.

The largest cluster, Cluster 3, contained 32 images, primarily representing transportation-related objects.

A rule-based music selection algorithm identified automobiles as the dominant category and selected *Danza Kuduro* through an energetic music mapping.

The results demonstrate how image feature extraction, unsupervised clustering, and algorithmic music selection can work together to support the creation of a themed video album.
