import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from clustering import reduce_dimensions

def plot_results(pca_df, dissimilar_pca_df, dissimilar_texts):
    # Buraya ekleyeceğiniz kod:
    dissimilar_texts = [f"{text} ({i+1})" for i, text in enumerate(dissimilar_texts)]

    fig = px.scatter(
        pca_df,
        x='Dim1',
        y='Dim2',
        color='kmeans_labels',
        labels={'Dim1': 'Dim1', 'Dim2': 'Dim2', 'kmeans_labels': 'Cluster'},
        color_continuous_scale='Viridis'
    )

    fig.add_trace(go.Scatter(
        x=dissimilar_pca_df['Dim1'],
        y=dissimilar_pca_df['Dim2'],
        mode='markers+text',
        text=dissimilar_texts,  # Güncellenmiş dissimilar_texts kullanılıyor
        textposition='top center',
        marker=dict(size=12, color='red', symbol='diamond'),
        name='Dissimilar Samples'
    ))

    fig.update_layout(
        hovermode='closest', 
        title='PCA ile Boyut İndirgeme ve K-Means Gruplama',
        xaxis_title='Dim1',
        yaxis_title='Dim2'
    )

    fig.show()

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=pca_df['Dim1'],
        y=pca_df['Dim2'],
        mode='markers',
        marker=dict(size=12, color=pca_df['kmeans_labels'], colorscale='Viridis'),
        hoverinfo='text',
        hovertext=pca_df["text"],
        name='Clusters'
    ))

    fig2.add_trace(go.Scatter(
        x=dissimilar_pca_df['Dim1'],
        y=dissimilar_pca_df['Dim2'],
        mode='markers+text',
        text=dissimilar_texts,  # Güncellenmiş dissimilar_texts kullanılıyor
        textposition='top right',
        marker=dict(size=12, color='red', symbol='diamond'),
        name='Dissimilar Samples',
        hoverinfo='text',
        hovertext=dissimilar_texts  # Güncellenmiş dissimilar_texts kullanılıyor
    ))

    fig2.update_layout(
        title='PCA ile Boyut İndirgeme ve K-Means Gruplama',
        xaxis_title='Dim1',
        yaxis_title='Dim2',
        hovermode='closest'
    )

    fig2.show()




def reduce_and_plot(embeddings, texts, cleaned_texts, kmeans_labels, dissimilar_indices, output_dir):
    print("Debug: Inside reduce_and_plot function")
    print(f"Debug: embeddings shape {embeddings.shape}")
    print(f"Debug: kmeans_labels length {len(kmeans_labels)}")
    print(f"Debug: dissimilar_indices {dissimilar_indices}")

    print("Reducing dimensions for plotting...")
    # Apply PCA again to reduce dimensions to 2 for plotting
    pca_result_2d = reduce_dimensions(embeddings, n_components=2)
    print(f"Debug: pca_result_2d shape {pca_result_2d.shape}")
    
    dissimilar_pca_df = pd.DataFrame(pca_result_2d[dissimilar_indices], columns=['Dim1', 'Dim2'])
    dissimilar_pca_df['text'] = [cleaned_texts[i] for i in dissimilar_indices]

    # Save the most dissimilar points as .txt files
    for i in dissimilar_indices:
        original_filename = texts[i]['file_name']
        output_path = os.path.join(output_dir, original_filename)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(texts[i]['text'])
        print(f"Saved {original_filename} to {output_path}")

    pca_df = pd.DataFrame(pca_result_2d, columns=['Dim1', 'Dim2'])
    pca_df['kmeans_labels'] = kmeans_labels
    pca_df['text'] = cleaned_texts

    plot_results(pca_df, dissimilar_pca_df, dissimilar_pca_df['text'])
