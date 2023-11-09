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

@app.route('/search', methods=['POST'])
def search():
    print(request.form)
    song_name = request.form.get('song_name')
    year = request.form.get('year')
    
    # Call your search function and process the input values
    result = search_song(song_name, year)
    
    return jsonify({'result': result})

@app.route('/recommendations', methods=['POST'])
def recommendations():
    song_name = request.form.get('song_name')
    year = request.form.get('year')
    num_recommendations = request.form.get('num_recommendations')
    
    # Call your recommendations function and process the input values
    # recommendations = get_recommendations(song_name, year, num_recommendations)
    # return render_template('recommendations.html', recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
