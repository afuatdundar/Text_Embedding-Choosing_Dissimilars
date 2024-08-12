import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def reduce_dimensions(embeddings, n_components):
    pca_model = PCA(n_components=n_components)
    pca_result = pca_model.fit_transform(embeddings)
    return pca_result

def cluster_texts(pca_result, n_clusters=5):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans_model.fit_predict(pca_result)
    return kmeans_labels

def compute_cosine_distances(embeddings):
    return cdist(embeddings, embeddings, metric='cosine')

def most_dissimilar_subset(distance_matrix, top_n):
    n = distance_matrix.shape[0]
    selected_indices = [np.random.choice(n)]
    
    for _ in range(top_n - 1):
        remaining_indices = list(set(range(n)) - set(selected_indices))
        dist_to_selected = distance_matrix[remaining_indices][:, selected_indices]
        sum_dist_to_selected = np.sum(dist_to_selected, axis=1)
        next_index = remaining_indices[np.argmax(sum_dist_to_selected)]
        selected_indices.append(next_index)
    
    return selected_indices