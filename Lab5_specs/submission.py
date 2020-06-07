import pandas as pd
import numpy as np

np.set_printoptions(threshold=np.nan)
data_file='./asset/a'

#probablity rate 
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

#read in the data
raw_data = pd.read_csv(data_file, sep=',')
labels=raw_data['Label'].values
data=np.stack((raw_data['Col1'].values,raw_data['Col2'].values), axis=-1)
#fixed parameters
weights = np.zeros(3) 
num_epochs = 50000
learning_rate = 50e-5

def logistic_regression(data, labels, weights, num_epochs, learning_rate): # do not change the heading of the function
    dataM = np.insert(data,0,1,axis=1)
    labelM = np.mat(labels).T
    print(dataM)
    print(sigmoid(dataM*weights) - labelM)
    for i in range(num_epochs):
        #the difference between the real probablity and label/error rate
        error = sigmoid(dataM*weights) - labelM
        weights = weights - learning_rate * dataM.T*error
    return np.array(weights.T.tolist()[0])

weights = logistic_regression(data,labels,weights,num_epochs,learning_rate)
print(weights)


   