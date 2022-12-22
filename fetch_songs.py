import os
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="7631e11aaaa7413e8cec2c52ec12bec1", client_secret="2e006a75e4d147149d864aeed88fcbd8"))
def find_song(name, year):
    """
    This function returns a dataframe with data for a song given the name and release year.
    The function uses Spotipy to fetch audio features and metadata for the specified song.
    """
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    # print(results)
    if results['tracks']['items'] == []: #song not found
        return None
    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    # print(audio_features)
    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value
    # print(pd.DataFrame(song_data))
    return pd.DataFrame(song_data)

def get_song_data(song, dataset):
    """
    Gets the song data for a specific song from dataset otherwise looks for song
    from spotify. The song argument takes the form of a dictionary with
    key-value pairs for the name and release year of the song.
    """
    try:
        song_data = dataset[(dataset['name'] == song['name'])
                                & (dataset['year'] == song['year'])].iloc[0]
        # print(song_data)
        return song_data
    except IndexError:
        return find_song(song['name'], song['year']) #find the song from spotify

def get_mean_vector(songs, dataset):
    """
    Compute the average vector of the audio
    and metadata features for each song the user has listened to.
    """
    song_vectors = []

    #get song data for each song listened to otherwise the song does not exist in dataset or on spotify
    for song in songs:
        song_data = get_song_data(song, dataset)
        if song_data is None:
            print('Sorry but the song {} does not exist in dataset or on spotify'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):
    """
    function for flattening a list of dictionaries.
    """
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict

def recommend_songs(song_list, dataset,cluster_song, n_songs=10):

    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, dataset)
    scaler = cluster_song.steps[0][1]
    scaled_data = scaler.transform(dataset[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = dataset.iloc[index]
    #Find the n-closest data points in the dataset (excluding the points
    #from the songs in the user’s listening history) to this average vector
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    print("################################## Song Recommendations ##############################")
    print(rec_songs[metadata_cols].to_dict(orient='records'))
    return rec_songs[metadata_cols].to_dict(orient='records')
