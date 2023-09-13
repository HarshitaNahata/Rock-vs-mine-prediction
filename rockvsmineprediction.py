import numpy as np
#numpy is for arrays
import pandas as pd
#pandas is for several data processing steps and loading data into dataframes
from sklearn.model_selection import train_test_split
#to split data in to training and testing data
from sklearn.linear_model import LogisticRegression
#to perform logistic regression we import logistic regression model
from sklearn.metrics import accuracy_score
#to find the accuracy of our model
#Data collection and Data processing
#importing the dataset to a pandas dataframe
sonar_data = pd.read_csv('sonar_data.csv',header=None)
sonar_data.head()
#displays first five rows of dataset
sonar_data.shape
#to see the number of rows and columns
sonar_data.describe()
#gives us mean,standard deviation and other parameters too which means statistical measures
sonar_data[60].value_counts()
#number of examples for rocks and mines,rock and mine is at 60th column
sonar_data.groupby(60).mean()
#we get mean value for all the columns
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
print(X)
print(Y)
#to separate data and labels
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
#to split the data into training and test data
#x_train is training data and x_test is the testing data
#y_train is the label for training data and y_test is the label for testing data
#test size is 0.1 which means we need 10% of data to be test data
#stratify is used to split data based on number of rock or mine 
#random state is to split data in particular order
print(X.shape,X_train.shape,X_test.shape)
#to find the number of training and testing data
#MODEL TRAINING by logistic regression
print(X_train)
print(Y_train)
#WE PRINT the training data and data labels
model=LogisticRegression()
#training the logistic regression model with the training data
model.fit(X_train,Y_train)
#we include training data and training label
#we feed this data to train our machine learning model
#MODEL EVALUATION
#we find the accuracy of the model on training data and always accuracy of training data is high because the model has seen the training data and not the test data
#any accuracy greater than 70% is good
X_train_prediction=model.predict(X_train)
accuracy_of_training_data=accuracy_score(X_train_prediction,Y_train)
#X_train_prediction is the prediction our model makes based on its learning and Y_train is the answer to those predictions
print('Accuracy of training data:',accuracy_of_training_data)
#we find the accuracy of testing data
X_test_prediction=model.predict(X_test)
accuracy_of_testing_data=accuracy_score(X_test_prediction,Y_test)
print('Accuracy of testing data:',accuracy_of_testing_data)
#Predictive system that can predict whether the object is rock or mine using sonar data
input_data=(0.0311,0.0491,0.0692,0.0831,0.0079,0.0200,0.0981,0.1016,0.2025,0.0767,0.1767,0.2555,0.2812,0.2722,0.3227,0.3463,0.5395,0.7911,0.9064,0.8701,0.7672,0.2957,0.4148,0.6043,0.3178,0.3482,0.6158,0.8049,0.6289,0.4999,0.5830,0.6660,0.4124,0.1260,0.2487,0.4676,0.5382,0.3150,0.2139,0.1848,0.1679,0.2328,0.1015,0.0713,0.0615,0.0779,0.0761,0.0845,0.0592,0.0068,0.0089,0.0087,0.0032,0.0130,0.0188,0.0101,0.0229,0.0182,0.0046,0.0038)
#to convert the datatype into a numpy array
input_data_as_numpy_array =np.asarray(input_data)
#we reshape the data as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#there is 1 instance and we predict one label for that instance
prediction=model.predict(input_data_reshaped)
#we store the trained logistic regression model in the variable called prediction and model.predict either return R or M
print(prediction)
if(prediction[0]=='R'):
    print('The object is a rock')
else:
    print('The object is a mine')