# IMPORTS
import streamlit as st
import numpy as np
import pandas as pd
#modeule to load saved models
from tensorflow.keras.models import load_model  

#Preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from scipy.sparse import lil_matrix

#Scoring Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from keras.models import Sequential
from keras.layers import *

# LOAD DATA
# path to the dataset
train_data = pd.read_csv('./data/trainingData.csv')
train_data.head()


# LOADING MODELS
# @st.cache(allow_output_mutation=True)
# def load_models():
    # Predicting Latitude and Longitude (LAT)
model_3 = load_model('./models/model_3.h5')
# Predicting Relative Position (RP)
model_rp = load_model('./models/model_rp.h5')



# # PREDICTING
# # Predicting Latitude and Longitude (LAT)
# def predict_coordinate(x_test, y_test): 
#     predict_coordinates = model_3.predict(x_test)
#     predict_coordinates = np.argmax(predict_coordinates,axis=1) # raw probabilities to choose class (highest probability)
#     y_true= np.argmax(y_test,axis=1) 
#     sac1 = metrics.accuracy_score(y_true, predict_coordinates)
#     print("Accuracy score: {}".format(sac1))


# # Predicting Relative Position (RP)
# predict_position = model_rp.predict(x_test1)

# predict_position = np.argmax(predict_position,axis=1) # raw probabilities to choose class (highest probability)

# y_true= np.argmax(y_test1,axis=1) 
# sac1 = metrics.accuracy_score(y_true, predict_position)
# print("Accuracy score: {}".format(sac1))

def welcome():
	return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs

# # PREDICTING
LATITUDE =""
LONGITUDE =""
# # Predicting Latitude and Longitude (LAT)
def predict_coordinates(x_output):
	coordinates = model_3.predict([[x_output]])
	print(coordinates)
	return coordinates
# # Predicting Relative Position (RP)    
# def predict_relative_oposition(x_output, y_output, x_rel_pos_input, y_rel_pos_input):
# 	relative_oposition = model_rp.predict([[x_output, y_output, x_rel_pos_input, y_rel_pos_input]])
# 	print(relative_oposition)
# 	return relative_oposition

	

# this is the main function 
def main():
	st.title("Loacalization and Prediction using DeepLearning")
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;"> Loacalization and Prediction using DeepLearning </h1>
	</div>
	"""
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)
	
	
	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
	x_output = st.text_input("INPUT_X", "")
	y_output = st.text_input("INPUT_Y", "")
	# x_rel_pos_input = st.text_input("INPUT_REL_X", "Type Here")
	# y_rel_pos_input = st.text_input("INPUT_REL_Y", "Type Here")
        
	
	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	if st.button("Predict"):
            LATITUDE = predict_coordinates(x_output)
            # LONGITUDE = predict_coordinates(x_output, y_output, x_rel_pos_input, y_rel_pos_input)
            st.success('The LATITUDE is {} & The LONGITUDE is {}'.format(LATITUDE, LONGITUDE))
            st.success('The LATITUDE is {} & The LONGITUDE is {}'.format(LATITUDE, LONGITUDE))
	
if __name__=='__main__':
	main()