import os
import re
import string
import numpy as np
import pandas as pd
import nltk
from clustering import reduce_dimensions, cluster_texts, compute_cosine_distances, most_dissimilar_subset
from data_processing import read_texts, preprocess_texts
from embedding import compute_embeddings
from plotting import reduce_and_plot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

nltk.download('punkt')

def plot_explained_variance(embeddings):
    pca = PCA().fit(embeddings)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.axhline(y=0.90, color='r', linestyle='-')
    plt.axhline(y=0.95, color='g', linestyle='-')
    plt.grid(True)
    plt.show()

import os

def main():
    # Define paths relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")
    directory = os.path.join(script_dir, "samples")

    print("Starting data processing...")
    # Metin dosyalarını okuma ve ön işleme
    texts = read_texts(directory)
    cleaned_texts = preprocess_texts(texts)
    print("Data processing completed.")

    print("Starting embedding computation...")
    # Embedding hesaplama
    embeddings = compute_embeddings(cleaned_texts)
    print("Embedding computation completed.")

    print("Plotting explained variance to determine optimal number of components...")
    # Plot explained variance to determine optimal number of components
    plot_explained_variance(embeddings)

    n_components = int(input("Enter the number of PCA components: "))

    print("Applying PCA...")
    # Boyut indirgeme (PCA kullanarak)
    pca_result = reduce_dimensions(embeddings, n_components=n_components)
    print("PCA completed.")

    print("Computing distance matrix...")
    # scipy ile cosine distance matrisini hesaplama
    distance_matrix = compute_cosine_distances(pca_result)
    print("Distance matrix computed.")

    print("Selecting most dissimilar points...")
    # En farklı metinleri seç
    top_n = 10
    dissimilar_indices = most_dissimilar_subset(distance_matrix, top_n)
    print("Selection of diverse points completed.")

    print("Clustering data...")
    # K-Means ile gruplama
    kmeans_labels = cluster_texts(pca_result, n_clusters=5)
    print("Clustering completed.")

    # Call the reduce_and_plot function from plotting.py
    reduce_and_plot(embeddings, texts, cleaned_texts, kmeans_labels, dissimilar_indices, output_dir)

if __name__ == "__main__":
    main()
