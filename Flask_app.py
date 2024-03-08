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


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# load the dataset
dataframe = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()