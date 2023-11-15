from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from packages.search_song import search_song
from packages.run_recommender import get_feature_vector, show_similar_songs


# load data
dat = pd.read_csv('./data/processed/dat_for_recommender.csv')

song_features_normalized = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness']
song_features_not_normalized = ['duration_ms', 'key', 'loudness', 'mode', 'tempo']

all_features = song_features_normalized + song_features_not_normalized + ['decade', 'popularity']

#app
app = Flask(__name__,template_folder='templates',static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

 
 
 
@app.route('/search', methods=['GET'])
def search():
    print(request.method)
    if request.method == 'GET':
        return render_template('search.html')
    
@app.route('/search_result', methods=['POST'])
def search_result():
    print(request.form)
    song_name = request.form.get('song_name')
    year = request.form.get('year')
    
    # Call your search function and process the input values
    found_flag, found_song  = search_song(song_name,dat)
    if found_flag:
            print("Perfect, this song is in the dataset:")
            
            for i in range(len(found_song)):
                print("Song Name: " + found_song[i][0])
                print("Artist(s): " + found_song[i][1])
                print("Release Date: " + found_song[i][2])
    else:
        print("Sorry, this song is not in the dataset. Please try another song!")
    return render_template("search_result.html", data=found_song) 

@app.route('/recommendations', methods=['GET'])
def recommendations():
    print(request.method)
    if request.method == 'GET':
        return render_template('recommendations.html')

@app.route('/recommendations_result', methods=['POST'])
def recommendations_result():
    song_name = request.form.get('song_name')
    year = int(request.form.get('year'))
    num_recommendations = request.form.get('num_recommendations')
     
    similar_songs = show_similar_songs(song_name, year, dat, all_features, num_recommendations, plot_type='wordcloud')
    # Call your recommendations function and process the input values
    # recommendations = get_recommendations(song_name, year, num_recommendations)
    # return render_template('recommendations.html', recommendations=recommendations)

    return render_template("recommendations_result.html",data = similar_songs) 


if __name__ == "__main__":
    app.run(debug=True)
