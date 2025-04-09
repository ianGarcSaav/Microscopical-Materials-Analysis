import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def perform_kmeans_clustering(csv_path, n_clusters=3, random_state=42):
    """
    Performs K-Means clustering on the data in the given CSV file.

    Args:
        csv_path (str): Path to the CSV file containing the measurements.
        n_clusters (int): The number of clusters to form.
        random_state (int): Random state for reproducibility.

    Returns:
        dict: A dictionary mapping 'Label' to 'Cluster'.
              Returns None if there's an error.
    """
    try:
        # Read the data from the CSV file
        df = pd.read_csv(csv_path)

        # Select the features for clustering (all columns except 'Label')
        features = df.columns[1:]  # Exclude the 'Label' column
        X = df[features]

        # Scale the features using StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)  # Explicitly set n_init
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # Return the cluster labels as a dictionary
        cluster_labels = df[['Label', 'Cluster']].set_index('Label').to_dict()['Cluster']
        return cluster_labels

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"An error occurred during clustering: {e}")
        return None

def save_clustered_data(df, output_csv_path):
    """
    Saves the DataFrame with cluster labels to a new CSV file.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data and cluster labels.
        output_csv_path (str): The path to save the new CSV file.
    """
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Clustered data saved to {output_csv_path}")
    except Exception as e:
        print(f"An error occurred while saving the clustered data: {e}")

# Example usage (can be removed or commented out for library use)
if __name__ == '__main__':
    # Create a dummy CSV file for testing
    data = {'Label': [1, 2, 3, 4, 5],
            'Area': [10, 12, 23, 25, 31],
            'Perimeter': [3, 5, 7, 9, 11]}
    dummy_csv_path = 'dummy_measurements.csv'
    df = pd.DataFrame(data)
    df.to_csv(dummy_csv_path, index=False)

    # Example: Perform clustering and save the results
    cluster_labels = perform_kmeans_clustering(dummy_csv_path, n_clusters=2)
    if cluster_labels is not None:
        print(cluster_labels)
