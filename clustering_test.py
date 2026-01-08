import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# 1. Artificial Data Create Karna (Testing ke liye)
# Hum "Moons" shape data use kar rahe hain kyunke DBSCAN yahan K-Means se behtar kaam karta hai
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])

# 2. Data Scaling (Clustering ke liye boht zaroori hai)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# --- K-MEANS ALGORITHM ---
# K-Means centroids use karta hai (Spherical clusters ke liye best hai)
kmeans = KMeans(n_clusters=2, random_state=42)
df['KMeans_Labels'] = kmeans.fit_predict(X_scaled)

# --- DBSCAN ALGORITHM ---
# DBSCAN density check karta hai (Irregular shapes aur outliers ke liye best hai)
dbscan = DBSCAN(eps=0.3, min_samples=5)
df['DBSCAN_Labels'] = dbscan.fit_predict(X_scaled)

# 3. Visualizing the Results
plt.figure(figsize=(12, 5))

# K-Means Plot
plt.subplot(1, 2, 1)
plt.scatter(df['Feature_1'], df['Feature_2'], c=df['KMeans_Labels'], cmap='viridis')
plt.title("K-Means (Centroid Based)")

# DBSCAN Plot
plt.subplot(1, 2, 2)
plt.scatter(df['Feature_1'], df['Feature_2'], c=df['DBSCAN_Labels'], cmap='plasma')
plt.title("DBSCAN (Density Based)")

print("✅ Clustering Complete! Displaying Plots...")
plt.show()

# Results Preview
print("\nFirst 5 rows with cluster labels:")
print(df.head())