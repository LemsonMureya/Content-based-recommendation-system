import os
from fetch_songs import find_song, get_song_data, recommend_songs
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

#Read data
# print(data.info())
# print(genre_data.info())
# print(year_data.info())

# K-means clustering algorithm is used to divide the genres in this dataset
# into ten clusters based on the numerical audio features of each genres.
def k_means_with_genre(genre):
    cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
    X = genre.select_dtypes(np.number)
    cluster_pipeline.fit(X)
    genre['cluster'] = cluster_pipeline.predict(X)

    #easily visualize the genre clusters in a two-dimensional
    #coordinate plane by using Plotly’s scatter function
    #to compress the data into a two-dimensional space using t-SNE
    tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
    genre_embedding = tsne_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
    projection['genres'] = genre['genres']
    projection['cluster'] = genre['cluster']
    #generates an html visaulization/scatter plot of clusters by genre
    fig = px.scatter(
        projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
    fig.show()

# cluster the songs using K-means with PCA
def k_means_with_song(song_data):
    print("Clustering Songs with K-Means using PCA")
    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=20,
                                   verbose=False))
                                 ], verbose=False)
    X = song_data.select_dtypes(np.number)
    number_cols = list(X.columns)
    song_cluster_pipeline.fit(X)
    song_cluster_labels = song_cluster_pipeline.predict(X)
    song_data['cluster_label'] = song_cluster_labels

    #The song data frame is much larger than the genre data frame so I decided to use PCA for dimensionality reduction rather
    #than t-SNE because it runs significantly faster
    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
    projection['title'] = song_data['name']
    projection['cluster'] = song_data['cluster_label']

    fig = px.scatter(
        projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
    fig.show()
    return song_cluster_pipeline

def main():
#read csv data
#change the path on your computer to access the files
    data = pd.read_csv('C:/Users/Lemson Mureya/Desktop/FinalProjML/data.csv')
    genre_data = pd.read_csv('C:/Users/Lemson Mureya/Desktop/FinalProjML/data_by_genres.csv')
    year_data = pd.read_csv('C:/Users/Lemson Mureya/Desktop/FinalProjML/data_by_year.csv')

    print("######################################################################################")
    print("Clustering Genres with K-Means using TSNE")
    k_means_with_genre(genre_data)

    print("######################################################################################")
    # k_means_with_song(data) #generates an html visualization/scatter plot of clusters by songs
    # find_song('Confession', '2021')
    # get_song_data({'name': 'Confession', 'year':2021}, data)
    clusters = k_means_with_song(data)
    recommend_songs([{'name': 'Famous', 'year':2016},
                {'name': 'All Falls Down', 'year': 2004},
                {'name': 'Good Life', 'year': 2007},
                {'name': 'Donda', 'year': 2021},
                {'name': 'Champion', 'year': 2007}],  data, clusters)
if __name__=='__main__':
    main()
