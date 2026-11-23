import numpy as np
from sklearn.cluster import KMeans

# Load the feature matrix previously extracted using ResNet-18.
features = np.load("features.npy")

# Display the dimensions of the loaded feature matrix.
print(f"Feature matrix shape: {features.shape}")

# Set the number of clusters we want K-Means to create.
NUM_CLUSTERS = 10

# Initialize K-Means with 10 clusters and a fixed random seed.
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)

# Analyze the feature vectors and assign each image to a cluster.
cluster_labels = kmeans.fit_predict(features)

# Display the cluster assigned to each image.
print(f"Cluster assignments: {cluster_labels}")

# Count how many images were assigned to each cluster.
for cluster_id in range(NUM_CLUSTERS):
    count = np.sum(cluster_labels == cluster_id)
    print(f"Cluster {cluster_id}: {count} images")

# Save each image's assigned cluster ID for later visualization and video generation.
np.save("cluster_labels.npy", cluster_labels)

