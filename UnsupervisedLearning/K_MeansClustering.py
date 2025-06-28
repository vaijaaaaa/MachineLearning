import numpy as np
import matplotlib.pyplot as plt

# Generate random 2D data points
np.random.seed(42)
points1 = np.random.randn(50, 2) + np.array([2, 2])
points2 = np.random.randn(50, 2) + np.array([7, 7])
data = np.vstack((points1, points2))

# Number of clusters
K = 2

# Step 1: Initialize centroids randomly from data points
centroids = data[np.random.choice(len(data), K, replace=False)]

# Function to compute Euclidean distance
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Step 2-4: Iterate until convergence (or fixed steps)
for iteration in range(10):  # Fixed 10 steps
    # Step 2: Assign each point to nearest centroid
    clusters = [[] for _ in range(K)]
    for point in data:
        distances = [euclidean(point, centroid) for centroid in centroids]
        closest_index = np.argmin(distances)
        clusters[closest_index].append(point)

    # Step 3: Update centroids
    new_centroids = []
    for cluster in clusters:
        new_centroids.append(np.mean(cluster, axis=0))
    centroids = np.array(new_centroids)

# Step 5: Plot final clusters
colors = ['red', 'blue', 'green']
for i in range(K):
    cluster_points = np.array(clusters[i])
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering (from Scratch)")
plt.legend()
plt.grid(True)
plt.show()
