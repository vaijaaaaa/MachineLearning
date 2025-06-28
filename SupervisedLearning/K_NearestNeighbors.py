import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r"C:\Users\vaijnath\Desktop\MachineLearning\SupervisedLearning\knn_data.csv")
X = data['Hours'].values
Y = data['Result'].values

# Combine into list of (x, y) tuples
dataset = list(zip(X, Y))

# Distance function (Euclidean)
def euclidean_distance(a, b):
    return abs(a - b)

# KNN function
def knn_predict(data, query_point, k=3):
    # Calculate all distances
    distances = [(euclidean_distance(query_point, x), label) for x, label in data]
    # Sort by distance
    distances.sort(key=lambda tup: tup[0])
    # Pick k closest labels
    k_nearest_labels = [label for _, label in distances[:k]]
    # Majority vote
    prediction = Counter(k_nearest_labels).most_common(1)[0][0]
    return prediction

# Predict for a new value
query = 8
k = 3
prediction = knn_predict(dataset, query, k)
print(f"\n📚 Predicted class for {query} hours (k={k}): {prediction} (0=Fail, 1=Pass)")

# Optional: visualize
plt.scatter(X, Y, c=Y, cmap='bwr', s=100, label='Training Data')
plt.axvline(x=query, color='green', linestyle='--', label=f'Query: {query} hrs')
plt.title(f'KNN Classification (k={k})')
plt.xlabel("Hours Studied")
plt.ylabel("Pass/Fail")
plt.yticks([0, 1])
plt.legend()
plt.grid(True)
plt.show()
