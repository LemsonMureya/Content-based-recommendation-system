# Content-based-recommendation-system

## About
Program is a content-based recommendation system using Spotify data to personalize user experience and recommend accurate and relevant content. 
The general structure of the files include: 
* 'recommendation.py': contains the main function and algorithm.  
* 'fetch_songs.py' : uses Spotify’s web API to retrieve audio features and metadata about songs such as the song’s acousticness, popularity, tempo, loudness, and the year in which it was released. 
* This data is used to build a music recommendation system that recommends songs to users based on both the audio features and the metadata of the songs that they have listened to.

## Data Sources
* Kaggle
: Spotify Dataset, which is available on Kaggle and contains metadata and audio features for over 100,000 different songs. Use data files from this dataset, data.csv, which contains data for individual songs, data_by_genre, which contains the data grouped by the genres, and data_by_year, which contains years in which the songs were released.
* Spotipy API :
fetch data for songs that did not exist in Kaggle’s Spotify Song Dataset.

## Requirements
Before running the program, make sure to import all the libraries and modules:
* numpy as np
* pandas as pd
* seaborn as sns
* plotly.express as px
* matplotlib.pyplot as plt
* import warnings

## Program Flow
* User inputs name(s) of song and year in which it was released (this is acting as user history) and the program outputs the desired recommendations  
* Program outputs a dictionary of recommended songs with artist name, song title and year in whcu it was released.

## Improvements
* A more appealing UI
* A hybrid model utilizing both content-based filtering and collaborative filtering to enable diverse recommendations




