import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('basketball_training_dataset.csv')


# Function to prepare data for clustering
def prepare_data(df):
    # Remove any rows with missing values
    df_clean = df.dropna()

    # Get numerical columns only
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    X = df_clean[numeric_cols]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df_clean, numeric_cols


# Function to determine optimal number of clusters using elbow method
def find_optimal_clusters(X, max_clusters=10):
    inertias = []

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Plot elbow curve and save to file
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('elbow_plot.png')
    plt.close()

    print("Elbow plot has been saved as 'elbow_plot.png'")
    return inertias


# Function to perform clustering and visualize results
def perform_clustering(X, df_clean, n_clusters, numeric_cols):
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Add cluster labels to the dataframe
    df_clean['Cluster'] = clusters

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create visualization and save to file
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Cluster Visualization using PCA')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('cluster_visualization.png')
    plt.close()

    print("Cluster visualization has been saved as 'cluster_visualization.png'")

    # Calculate and display cluster characteristics
    cluster_stats = df_clean.groupby('Cluster')[numeric_cols].mean()
    print("\nCluster Characteristics:")
    print(cluster_stats)

    # Save cluster statistics to CSV
    cluster_stats.to_csv('cluster_statistics.csv')
    print("Cluster statistics have been saved as 'cluster_statistics.csv'")

    return df_clean, cluster_stats


# Main execution
def main():
    # Prepare the data
    X_scaled, df_clean, numeric_cols = prepare_data(df)

    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    inertias = find_optimal_clusters(X_scaled)

    # Let's assume we chose n_clusters based on elbow method
    n_clusters = 4  # This can be adjusted based on the elbow plot

    # Perform clustering
    print(f"\nPerforming clustering with {n_clusters} clusters...")
    df_clustered, cluster_stats = perform_clustering(X_scaled, df_clean, n_clusters, numeric_cols)

    # Save results
    df_clustered.to_csv('basketball_clusters.csv', index=False)
    print("\nClustered data has been saved to 'basketball_clusters.csv'")


if __name__ == "__main__":
    main()