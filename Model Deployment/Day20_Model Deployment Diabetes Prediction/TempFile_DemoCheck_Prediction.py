# -*- coding: utf-8 -*-
"""
TempFile DemoCheck Prediction
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Users/manis/OneDrive/Desktop/30ML Challenge/Model Deployment/Day20_Model Deployment Diabetes Prediction/LogisticRegression_model.pkl', 'rb'))


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')